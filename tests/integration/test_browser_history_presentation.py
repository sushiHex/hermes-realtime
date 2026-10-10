"""Synthetic headless layout coverage of the production browser history paths."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

try:
    from playwright import sync_api as playwright
except ModuleNotFoundError:
    playwright = None
ROOT = Path(__file__).resolve().parents[2]


def _unavailable(message: str) -> None:
    if os.environ.get("HERMES_REALTIME_BROWSER_SELF_ACCEPTANCE") == "1":
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module")
def browser_script(tmp_path_factory: pytest.TempPathFactory) -> str:
    if (
        shutil.which("node") is None
        or not (ROOT / "web/node_modules/esbuild/bin/esbuild").is_file()
    ):
        _unavailable("Node and installed web dependencies are required for browser layout coverage")
    # The suffix exposes the real projection/highlight functions to synthetic inputs;
    # it does not replace any production code or lifecycle/scroll policy.
    source = (ROOT / "web/src/main.ts").read_text(encoding="utf-8")
    source += """
    globalThis.historyFixture = {
      project: projectPublicEvent,
      partial: updatePartialTranscript,
      startKaraoke: () => {
        window.project('assistant_text_generated',
          {turnId:'turn_karaoke',turnGeneration:1,segmentId:'segment_karaoke',
          text:'one two three'});
        const timing = parseSpeechTiming({turnId:'turn_karaoke',presentationTurnId:'turn_karaoke',
          turnGeneration:1,chunkId:'chunk_karaoke',segmentId:'segment_karaoke',streamId:'stream_karaoke',
          sampleRate:8000,timingSource:'estimated',timings:'0,3,0,8000;4,7,8000,16000;8,13,16000,24000'});
        streamMediaStarts.beginPassage('chunk_karaoke',0);
        streamMediaStarts.observePassage('chunk_karaoke',0.01);
        remoteAudio.dataset.streamId='stream_karaoke';
        Object.defineProperty(remoteAudio,'currentTime',{value:0.2,writable:true,configurable:true});
        applySpeechTiming(timing);
      },
      advanceKaraoke: (time) => { remoteAudio.currentTime = time; },
      historyAccounting: () => ({retainedRows:transcriptRetention.size,
        retainedCost:transcriptRetention.cost,taskViews:taskCardViews.size}),
      stopPersistentSession: async () => {
        setCredential({version:1,url:'wss://livekit.test',roomName:'synthetic-room',
          participantIdentity:'browser_0123456789abcdef',workerIdentity:'worker_hermes_browser',
          expiresInSeconds:60,token:'synthetic.token.value'});
        controller.beginConnect(); controller.bootstrapReady();
        window.fetch=async () => new Response('{}',{status:200});
        await stop();
      },
    };
    """
    target = tmp_path_factory.mktemp("history-browser") / "fixture.js"
    subprocess.run(
        [
            "node",
            "../node_modules/esbuild/bin/esbuild",
            "--bundle",
            "--loader=ts",
            "--format=iife",
            "--target=es2022",
            f"--outfile={target}",
            "--resolve-extensions=.ts,.js",
        ],
        input=source,
        text=True,
        encoding="utf-8",
        check=True,
        capture_output=True,
        cwd=ROOT / "web/src",
    )
    return target.read_text(encoding="utf-8")


@pytest.fixture
def page(browser_script: str):
    if playwright is None:
        _unavailable("install the browser-acceptance extra for browser layout coverage")
    required = os.environ.get("HERMES_REALTIME_BROWSER_SELF_ACCEPTANCE") == "1"
    executable = None
    if required:
        from tests.integration.test_browser_self_acceptance import _system_chrome

        executable = str(_system_chrome())
    with playwright.sync_playwright() as engine:
        try:
            browser = engine.chromium.launch(
                headless=True,
                executable_path=executable,
                ignore_default_args=["--hide-scrollbars"],
            )
        except playwright.Error as error:
            if not required and "Executable doesn't exist" in str(error):
                pytest.skip("install a Playwright Chromium browser for optional layout coverage")
            raise
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.set_content((ROOT / "web/index.html").read_text(encoding="utf-8"))
        page.add_style_tag(content=(ROOT / "web/src/styles.css").read_text(encoding="utf-8"))
        page.evaluate("""() => {
          window.fetch = async () => new Response('{}', {status: 503});
          navigator.mediaDevices = {enumerateDevices: async () => []};
        }""")
        page.add_script_tag(content=browser_script)
        page.evaluate("""() => {
          window.eventIndex = 0;
          window.project = (kind, data) => historyFixture.project({
            sequence: ++eventIndex, kind, data, monotonicMs: eventIndex * 1000,
          });
          window.chat = (count) => {
            for (let i = 0; i < count; i++)
              project('user_transcript', {text: 'Synthetic history line ' + i});
          };
          window.task = (id, status = 'active') => project('task_state', {taskId: id, status});
        }""")
        yield page
        browser.close()


def test_result_identity_replay_and_truthful_assistant_history(page) -> None:
    page.evaluate("""() => {
      task('task_first'); task('task_second');
      const text = 'Worker details ' + 'long synthetic detail '.repeat(35);
      project('task_result', {taskId: 'task_first', status: 'completed', text});
      project('task_result', {taskId: 'task_first', status: 'completed', text});
      project('task_result', {taskId: 'task_second', status: 'completed', text});
      const speech = {turnId:'turn_summary',turnGeneration:1,segmentId:'segment_summary',
        text:'Short spoken summary'};
      project('assistant_text_generated',speech);
      project('transcript_final',{...speech,role:'assistant'});
      project('assistant_turn_interrupted', {turnId:'turn_summary',turnGeneration:1});
    }""")
    assert page.locator('[data-role="task-result"]').count() == 0
    assert page.locator('[data-task-id="task_first"] details.task-result').count() == 1
    assert page.locator('[data-task-id="task_second"] details.task-result').count() == 1
    page.locator('[data-task-id="task_first"] summary').click()
    assert (
        "long synthetic detail" in page.locator('[data-task-id="task_first"] details').inner_text()
    )
    assert page.locator('[data-role="assistant"]').inner_text().endswith("Short spoken summary")
    assert page.locator('[data-role="assistant"]').get_attribute("data-interrupted") == "true"


def test_all_updates_preserve_manual_history_position_and_resume_at_live_end(page) -> None:
    page.evaluate("""() => {
      for (let i = 0; i < 40; i++) {
        project('transcript_final', {role:'assistant',text: 'Synthetic assistant line ' + i});
      }
      const scroll = document.querySelector('.conversation-scroll');
      scroll.scrollTop = 100;
      scroll.dispatchEvent(new Event('scroll'));
    }""")
    initial = page.locator(".conversation-scroll").evaluate("e => e.scrollTop")
    page.evaluate("""() => {
      historyFixture.partial('assistant', 'Synthetic incoming tokens');
      task('task_manual');
      project('task_result', {taskId:'task_manual', status:'completed', text:'Synthetic result'});
      project('transcript_final', {role:'assistant',text:'New speech continues'});
    }""")
    assert page.locator(".conversation-scroll").evaluate("e => e.scrollTop") == initial
    assert page.evaluate("window.scrollY") == 0
    page.evaluate("""() => {
      const scroll = document.querySelector('.conversation-scroll');
      scroll.scrollTop = scroll.scrollHeight;
      scroll.dispatchEvent(new Event('scroll'));
      project('transcript_final', {role:'assistant',text:'Follow this live update'});
    }""")
    assert (
        page.locator(".conversation-scroll").evaluate(
            "e => e.scrollHeight - e.clientHeight - e.scrollTop"
        )
        <= 2
    )


def test_active_cards_pin_only_after_crossing_top_stack_and_release_when_done(page) -> None:
    page.evaluate("""() => {
      for (let i=0; i<8; i++)
        project('transcript_final',{role:'assistant',text:'Earlier history ' + i});
      task('task_first');
      project('transcript_final',{role:'assistant',text:'Between active tasks'});
      task('task_second');
    }""")
    assert page.locator(".active-task-stack [data-task-id]").count() == 0
    page.evaluate("""() => {
      for (let i=0; i<30; i++)
        project('transcript_final',{role:'assistant',text:'Later history ' + i});
    }""")
    playwright.expect(page.locator(".active-task-stack [data-task-id]")).to_have_count(2)
    assert page.locator(".active-task-stack [data-task-id]").evaluate_all(
        "cards => cards.map(e => e.dataset.taskId)"
    ) == ["task_first", "task_second"]
    scroller_top = page.locator(".conversation-scroll").bounding_box()["y"]
    first_box = page.locator('[data-task-id="task_first"]').bounding_box()
    second_box = page.locator('[data-task-id="task_second"]').bounding_box()
    assert abs(first_box["y"] - scroller_top) <= 3
    assert second_box["y"] >= first_box["y"] + first_box["height"]
    page.evaluate("""() => {
      const scroll = document.querySelector('.conversation-scroll');
      scroll.scrollTop = 0; scroll.dispatchEvent(new Event('scroll'));
      task('task_first','cancelling');
    }""")
    assert page.locator(".active-task-stack [data-task-id]").count() == 2
    page.evaluate("task('task_first','completed')")
    assert page.locator('#transcript > [data-task-id="task_first"]').count() == 1
    assert page.locator(".active-task-stack [data-task-id]").count() == 1
    page.evaluate("task('task_second','failed')")
    assert page.locator(".active-task-stack [data-task-id]").count() == 0


def test_many_active_cards_have_bounded_focusable_overflow_and_survive_retention(page) -> None:
    page.evaluate("""() => {
      for (let i=0; i<9; i++) task('task_many_' + i);
      for (let i=0; i<150; i++)
        project('transcript_final',{role:'assistant',text:'Synthetic later line ' + i});
    }""")
    stack = page.locator(".active-task-stack")
    playwright.expect(stack.locator("[data-task-id]")).to_have_count(9)
    assert stack.get_attribute("role") == "region"
    assert stack.get_attribute("tabindex") == "0"
    assert stack.get_attribute("aria-label") == "Active background tasks"
    assert stack.evaluate("e => e.scrollHeight > e.clientHeight")
    panel_height = page.locator(".conversation-scroll").bounding_box()["height"]
    assert stack.bounding_box()["height"] <= panel_height * 0.4 + 1
    stack.focus()
    page.keyboard.press("End")
    playwright.expect(stack).to_be_focused()
    page.wait_for_function("document.querySelector('.active-task-stack').scrollTop > 0")


def test_running_pulse_is_absent_for_terminal_and_reduced_motion(page) -> None:
    page.evaluate("task('task_pulse')")
    card = page.locator('[data-task-id="task_pulse"]')
    assert card.evaluate("e => getComputedStyle(e).animationName") == "task-pulse"
    page.emulate_media(reduced_motion="reduce")
    assert card.evaluate("e => getComputedStyle(e).animationName") == "none"
    page.emulate_media(reduced_motion="no-preference")
    page.evaluate("task('task_pulse','completed')")
    assert card.evaluate("e => getComputedStyle(e).animationName") != "task-pulse"


def test_result_without_prior_state_and_replay_keeps_one_disclosure(page) -> None:
    page.evaluate("""project('task_result', {taskId:'task_orphan',status:'failed',
        text:'Synthetic failure details'})""")
    assert page.locator('[data-task-id="task_orphan"] details').count() == 1
    page.evaluate("""() => {
      task('task_orphan','failed');
      project('task_result', {taskId:'task_orphan',status:'failed',
        text:'Synthetic failure details'});
    }""")
    assert page.locator('[data-task-id="task_orphan"]').count() == 1
    assert page.locator('[data-task-id="task_orphan"] details').count() == 1


def test_cancellation_refusal_does_not_release_an_active_card(page) -> None:
    page.evaluate("""() => {
      task('task_refused');
      for(let i=0;i<30;i++)
        project('transcript_final',{role:'assistant',text:'Later synthetic line ' + i});
      task('task_refused','rejected');
    }""")
    assert page.locator('.active-task-stack [data-task-id="task_refused"]').count() == 1
    assert page.locator('[data-task-id="task_refused"]').get_attribute("data-status") == "active"


def test_active_state_replay_preserves_the_card_and_its_original_slot(page) -> None:
    page.evaluate("""() => {
      task('task_replay');
      for(let i=0;i<30;i++)
        project('transcript_final',{role:'assistant',text:'Later synthetic line ' + i});
      window.originalCard=document.querySelector('[data-task-id="task_replay"]');
      task('task_replay'); task('task_replay','cancelling');
    }""")
    assert page.locator('[data-task-id="task_replay"]').count() == 1
    assert page.locator(".task-history-slot").count() == 1
    page.evaluate("task('task_replay','interrupted')")
    assert page.locator(".task-history-slot").count() == 0
    assert page.locator('#transcript > [data-task-id="task_replay"]').evaluate(
        "e => e === window.originalCard"
    )


def test_active_stack_reflows_after_viewport_and_card_height_changes(page) -> None:
    page.evaluate("""() => {
      for(let i=0;i<9;i++) task('task_resize_' + i);
      for(let i=0;i<30;i++)
        project('transcript_final',{role:'assistant',text:'Later synthetic line ' + i});
      const card=document.querySelector('[data-task-id="task_resize_0"]');
      const extra=document.createElement('p');
      extra.textContent='Synthetic wrapping content '.repeat(30);
      card.append(extra);
    }""")
    page.wait_for_function("""() => {
      const card=document.querySelector('[data-task-id="task_resize_0"]');
      const anchor=document.querySelector('.task-history-slot');
      return Math.abs(card.getBoundingClientRect().height-anchor.getBoundingClientRect().height)<1;
    }""")
    page.set_viewport_size({"width": 1280, "height": 700})
    page.wait_for_function("""() => {
      const stack=document.querySelector('.active-task-stack');
      const scroll=document.querySelector('.conversation-scroll');
      return stack.getBoundingClientRect().height <= scroll.clientHeight * .4 + 1;
    }""")


@pytest.mark.parametrize("stop_path", ["event", "request"])
def test_persistent_session_stop_preserves_tasks_until_authoritative_completion(
    page,
    stop_path: str,
) -> None:
    page.evaluate("""() => {
      task('task_session_stop');
      for(let i=0;i<30;i++)
        project('transcript_final',{role:'assistant',text:'Later synthetic line ' + i});
      window.originalTask=document.querySelector('[data-task-id="task_session_stop"]');
    }""")
    if stop_path == "event":
        page.evaluate("project('session_stopped',{})")
    else:
        page.evaluate("historyFixture.stopPersistentSession()")
    assert page.locator(".active-task-stack [data-task-id]").count() == 1
    assert (
        page.locator('[data-task-id="task_session_stop"]').get_attribute("data-status") == "active"
    )
    page.evaluate("task('task_session_stop')")
    assert page.locator('[data-task-id="task_session_stop"]').count() == 1
    assert page.locator('[data-task-id="task_session_stop"]').evaluate(
        "e => e === window.originalTask"
    )
    page.evaluate("""project('task_result', {taskId:'task_session_stop',status:'completed',
      text:'Authoritative durable task completion'})""")
    assert page.locator(".active-task-stack [data-task-id]").count() == 0
    assert page.locator('#transcript > [data-task-id="task_session_stop"] details').count() == 1


def test_voice_clear_preserves_live_cards_and_anchors_until_exact_terminal_results(page) -> None:
    page.evaluate("""() => {
      task('task_deleted_terminal','completed');
      task('task_clear_pinned');
      for(let i=0;i<30;i++)
        project('transcript_final',{role:'assistant',text:'Voice history to delete ' + i});
      const scroll=document.querySelector('.conversation-scroll');
      scroll.scrollTop=0; scroll.dispatchEvent(new Event('scroll'));
      task('task_clear_future','cancelling');
      window.clearCards=['task_clear_pinned','task_clear_future'].map(id =>
        document.querySelector('[data-task-id="' + id + '"]'));
    }""")
    assert page.locator('.active-task-stack [data-task-id="task_clear_pinned"]').count() == 1
    assert page.locator('#transcript > [data-task-id="task_clear_future"]').count() == 1
    page.evaluate("project('voice_conversation_cleared',{})")
    assert page.locator("#transcript .task-history-slot").count() == 1
    assert page.locator('#transcript > [data-task-id="task_clear_future"]').count() == 1
    assert page.locator('[data-task-id="task_deleted_terminal"]').count() == 0
    assert page.locator('[data-role="assistant"]').count() == 0
    assert page.evaluate("historyFixture.historyAccounting()") == {
        "retainedRows": 0,
        "retainedCost": 0,
        "taskViews": 2,
    }
    page.evaluate("""() => {
      task('task_clear_pinned'); task('task_clear_future','cancelling');
    }""")
    assert page.evaluate("""clearCards.every(card => card.isConnected &&
      card === document.querySelector('[data-task-id="' + card.dataset.taskId + '"]'))""")
    assert page.locator('[data-operation="task"]').count() == 2
    page.evaluate("""() => {
      project('task_result',{taskId:'task_clear_pinned',status:'completed',
        text:'Pinned task result'});
      project('task_result',{taskId:'task_clear_future',status:'failed',
        text:'Future task result'});
    }""")
    assert page.locator(".active-task-stack [data-task-id]").count() == 0
    assert page.locator("#transcript > [data-task-id] details.task-result").count() == 2
    assert page.locator(".task-history-slot").count() == 0
    assert page.evaluate("""clearCards.every(card => card.parentElement.id === 'transcript' &&
      card === document.querySelector('[data-task-id="' + card.dataset.taskId + '"]'))""")
    assert page.evaluate("historyFixture.historyAccounting().retainedRows") == 2


def test_active_capacity_refuses_overflow_visibly_without_evicting_work(page) -> None:
    page.evaluate("""() => {
      for(let i=0;i<257;i++) task('task_capacity_' + i);
    }""")
    assert page.locator('[data-operation="task"]').count() == 256
    assert page.locator('[data-task-id="task_capacity_0"]').count() == 1
    assert page.locator('[data-task-id="task_capacity_256"]').count() == 0
    assert page.locator(".history-capacity-notice").inner_text() == (
        "Active task display capacity reached. Reconnect to refresh task state."
    )


def test_live_card_admission_preserves_exact_bounded_conversation_history(page) -> None:
    page.evaluate("""() => {
      for(let i=0;i<128;i++)
        project('transcript_final',{role:'assistant',text:'Synthetic history ' + i});
      task('task_retention');
    }""")
    retained = page.locator('#transcript > [data-role="assistant"]').evaluate_all(
        "rows => rows.map(e => e.childNodes[1].textContent)"
    )
    assert retained == [f"Synthetic history {i}" for i in range(128)]
    page.evaluate("task('task_retention','completed')")
    retained = page.locator('#transcript > [data-role="assistant"]').evaluate_all(
        "rows => rows.map(e => e.childNodes[1].textContent)"
    )
    assert retained == [f"Synthetic history {i}" for i in range(1, 128)]
    assert page.locator('#transcript > [data-task-id="task_retention"]').count() == 1


def test_real_karaoke_advances_without_stealing_history_or_page_position(page) -> None:
    page.evaluate("""() => {
      for(let i=0;i<40;i++)
        project('transcript_final',{role:'assistant',text:'Earlier synthetic line ' + i});
      historyFixture.startKaraoke();
    }""")
    playwright.expect(page.locator(".karaoke-active")).to_have_text("one")
    page.evaluate("""() => {
      const scroll=document.querySelector('.conversation-scroll');
      scroll.scrollTop=100; scroll.dispatchEvent(new Event('scroll'));
      historyFixture.advanceKaraoke(1.2);
    }""")
    playwright.expect(page.locator(".karaoke-active")).to_have_text("two")
    assert page.locator(".conversation-scroll").evaluate("e => e.scrollTop") == 100
    page.evaluate("historyFixture.advanceKaraoke(2.2)")
    playwright.expect(page.locator(".karaoke-active")).to_have_text("three")
    assert page.locator(".conversation-scroll").evaluate("e => e.scrollTop") == 100
    assert page.evaluate("window.scrollY") == 0


@pytest.mark.parametrize("gesture", ["wheel", "keyboard", "touch", "scrollbar"])
def test_user_scroll_gestures_suspend_following(page, gesture: str) -> None:
    page.evaluate("""() => {
      for(let i=0;i<40;i++)
        project('transcript_final',{role:'assistant',text:'Synthetic history ' + i});
    }""")
    # Coordinate input needs the native scrollbar's first painted frame; unlike
    # locator actions, raw mouse/CDP input has no actionability wait of its own.
    page.wait_for_function("""getComputedStyle(
      document.querySelector('#transcript > li:last-child')
    ).opacity === '1'""")
    scroll = page.locator(".conversation-scroll")
    if gesture == "wheel":
        scroll.hover()
        page.mouse.wheel(0, -600)
    elif gesture == "keyboard":
        scroll.evaluate("e => { e.tabIndex=0; e.focus(); }")
        page.keyboard.press("PageUp")
    elif gesture == "touch":
        box = scroll.bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + 60
        session = page.context.new_cdp_session(page)
        session.send("Emulation.setTouchEmulationEnabled", {"enabled": True})
        session.send(
            "Input.dispatchTouchEvent",
            {
                "type": "touchStart",
                "touchPoints": [{"x": x, "y": y}],
            },
        )
        for offset in (60, 120, 180, 240):
            session.send(
                "Input.dispatchTouchEvent",
                {
                    "type": "touchMove",
                    "touchPoints": [{"x": x, "y": y + offset}],
                },
            )
        session.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    else:
        box = scroll.evaluate("""e => {
          const box=e.getBoundingClientRect();
          const gutter=e.offsetWidth-e.clientWidth;
          return {x:box.right-gutter/2,y:box.top+box.height/3,gutter};
        }""")
        assert box["gutter"] > 0
        page.mouse.move(box["x"], box["y"])
        page.mouse.down()
        try:
            page.wait_for_function(
                """() => {
              const e=document.querySelector('.conversation-scroll');
              return e.scrollHeight-e.clientHeight-e.scrollTop > 100;
            }""",
                timeout=5000,
            )
        finally:
            page.mouse.up()
    page.wait_for_function("""() => {
      const e=document.querySelector('.conversation-scroll');
      return e.scrollHeight-e.clientHeight-e.scrollTop > 100;
    }""")
    # Wait for native keyboard smoothing/touch inertia rather than attributing
    # ongoing user motion to the subsequent production update.
    initial = scroll.evaluate("""e => new Promise(resolve => {
      let previous=e.scrollTop, stable=0;
      const frame=()=> {
        const current=e.scrollTop;
        stable=current===previous ? stable+1 : 0;
        previous=current;
        if(stable>=10) resolve(current);
        else requestAnimationFrame(frame);
      };
      requestAnimationFrame(frame);
    })""")
    page.evaluate("""project('transcript_final',
        {role:'assistant',text:'Live update during reading'})""")
    assert scroll.evaluate("e => e.scrollTop") == initial


@pytest.mark.parametrize("required", [False, True])
def test_missing_web_dependencies_fail_in_required_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
    required: bool,
) -> None:
    monkeypatch.setenv("HERMES_REALTIME_BROWSER_SELF_ACCEPTANCE", "1" if required else "0")
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    outcome = pytest.fail.Exception if required else pytest.skip.Exception
    with pytest.raises(
        (pytest.fail.Exception, pytest.skip.Exception), match="Node and installed web dependencies"
    ) as refused:
        browser_script.__wrapped__(tmp_path_factory)
    assert type(refused.value) is outcome


@pytest.mark.parametrize("required", [False, True])
def test_missing_playwright_fails_in_required_mode(
    monkeypatch: pytest.MonkeyPatch,
    required: bool,
) -> None:
    monkeypatch.setenv("HERMES_REALTIME_BROWSER_SELF_ACCEPTANCE", "1" if required else "0")
    monkeypatch.setitem(globals(), "playwright", None)
    outcome = pytest.fail.Exception if required else pytest.skip.Exception
    with pytest.raises(
        (pytest.fail.Exception, pytest.skip.Exception), match="browser-acceptance extra"
    ) as refused:
        next(page.__wrapped__("unused"))
    assert type(refused.value) is outcome
