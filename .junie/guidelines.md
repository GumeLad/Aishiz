# Project Guidelines

## Overview
Aishiz is an Android application written in Kotlin that aims to mimic the ChatGPT chat experience and layout. The app uses AndroidX, Material Components, ViewBinding, RecyclerView, and MotionLayout to deliver a modern, edge‑to‑edge chat UI with a message list and an input bar.

Primary goals:
- Provide a chat interface similar to the ChatGPT app, including a toolbar, scrollable message list, left/right aligned bubbles, and a bottom message composer.
- Clean architecture at the UI layer using `RecyclerView` and view types for different message roles (user vs assistant).
- Smooth insets handling (edge‑to‑edge) and Material theming.

## Repository structure (key files)

```
app/
  src/
    main/
      java/com/example/aishiz/
        MainActivity.kt            # Entry activity, sets up toolbar, RecyclerView, and send action
        ChatAdapter.kt             # RecyclerView adapter for chat messages
      res/
        layout/
          activity_main.xml        # Main screen layout (MotionLayout + AppBar + input bar)
          item_chat_message.xml    # Single chat message bubble layout
          recycler_view_item.xml   # (If used) Additional item layout
        layout-land/
          activity_main.xml        # Landscape variant of the main screen
        xml/
          activity_main_scene.xml  # MotionLayout scene
          activity_main_scene2.xml # MotionLayout scene (active)
        values/
          colors.xml, strings.xml, themes.xml
      AndroidManifest.xml
  build.gradle.kts (root and app module)
```

Note: Package declarations in Kotlin files currently use `package com.example.AiShiz`. Keep consistency if you add new Kotlin sources.

## Build and run

Preferred: Android Studio (Arctic Fox+ / latest recommended)
- Open the project folder `Aishiz`.
- Let Gradle sync complete; then Run the `app` configuration on an emulator or device.

Command line (Windows PowerShell or terminal in Android Studio):
```
./gradlew :app:assembleDebug
./gradlew :app:installDebug
```

## Tests

- Unit tests (JVM):
  - Location: `app/src/test/java/...`
  - Run: `./gradlew :app:testDebugUnitTest`

- Instrumented tests (device/emulator):
  - Location: `app/src/androidTest/java/...`
  - Run: `./gradlew :app:connectedDebugAndroidTest`

Junie should run relevant tests when changing Kotlin source that affects logic. For documentation‑only or resource‑only cosmetic changes (like pure Markdown updates), running tests is optional.

## Code style and guidelines

- Language: Kotlin. Follow standard Kotlin style and Android Kotlin style guides.
- ViewBinding: In Activities/Adapters, prefer ViewBinding over `findViewById` (already in use: `ActivityMainBinding`, `ItemChatMessageBinding`).
- Resources: Avoid hard‑coding colors/dimensions/strings in layouts or code; use `colors.xml`, `dimens.xml` (add if missing), and `strings.xml`.
- Theming: Use Material theme (themes.xml) and keep a consistent light/dark palette. Prefer theme attributes over raw colors in widgets.
- Edge‑to‑edge: Continue to use `WindowCompat.setDecorFitsSystemWindows(window, false)` and apply insets to paddings/margins where necessary.
- RecyclerView: Use stable IDs when feasible, and `LinearLayoutManager` vertical for chat.
- Packaging: Keep or reconcile the package name casing consistently across new files.

## UI/UX specification to mimic ChatGPT

Target layout characteristics:
- App bar: Title, optional actions (new chat, settings). Keep elevation low or flat; color follows Material theme.
- Message list: `RecyclerView` filling the space between the app bar and the input bar.
  - Two message roles: user (right‑aligned) and assistant (left‑aligned).
  - Bubbles with distinct colors:
    - Assistant: neutral surface (e.g., `surfaceContainer` or light grey), dark text.
    - User: primary color bubble with onPrimary text.
  - Optional avatar circle at the start of assistant messages; user messages may omit avatar.
  - Optional timestamp in a smaller, muted color.
  - Support multi‑line text; allow long content to wrap.
- Composer/input bar (bottom):
  - Text field with hint “Type a message…”.
  - Send button (icon preferred) enabled when text is non‑empty.
  - Optional: mic/attach actions.

Implementation notes:
- RecyclerView items should use two view types (USER vs ASSISTANT) with separate item layouts, e.g. `item_message_user.xml` and `item_message_assistant.xml`, or a single layout with conditional alignment via ConstraintLayout.
- Use shape drawables for rounded bubbles; avoid fixed pixel widths; respect paddings/margins typical of chat apps.
- Add a “typing indicator” row (three dots) as a distinct view type for better UX.
- Keep scrolling to the bottom when new messages arrive; consider `smoothScrollToPosition` for better feel.

Accessibility:
- All touchable controls must have `contentDescription` where applicable.
- Maintain sufficient color contrast in both light and dark modes.

Performance:
- Enable view recycling properly; avoid nested weights in item layouts; prefer ConstraintLayout for complex rows.
- Consider DiffUtil for large lists; for now, `notifyItemInserted` is acceptable for small demos.

## Junie workflow expectations

- For UI/layout changes (XML, drawables, themes) and Kotlin source edits:
  - Build the app before submitting changes: `./gradlew :app:assembleDebug`.
  - If Kotlin logic is changed, run unit tests. If UI behavior affecting instrumentation exists, consider `connectedDebugAndroidTest` when feasible.
- For documentation‑only updates (like this guidelines file):
  - No build or tests are required prior to submission unless requested.

## Next steps to reach a ChatGPT‑like UI

1. Define a `Message` model with `text: String` and `role: enum(User, Assistant)`.
2. Update `ChatAdapter` to use two view types and corresponding item layouts.
3. Replace any placeholder containers with an actual `RecyclerView` named `@+id/chat_recycler_view` in `activity_main.xml` if not already present.
4. Style bubbles with theme colors (`colorPrimary` for user, neutral surface for assistant) and rounded corners.
5. Add a typing indicator row and show it briefly after sending before the assistant response appears.
6. Optional: persist conversation state across configuration changes and process death (ViewModel + savedState / local DB).

These guidelines should keep the implementation consistent and aligned with the goal of mimicking the ChatGPT application UI and layout.
