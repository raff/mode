# Design System — MoDe

## Product Context
- **What this is:** A Morse code decoder that listens to live audio from SDRs/ham radios or decodes WAV files and outputs decoded text in real time
- **Who it's for:** Amateur radio operators, CW enthusiasts, SDR hobbyists
- **Space/industry:** Ham radio / SDR software tools (fldigi, WSJT-X, SDR#, Gqrx peer group)
- **Project type:** Desktop GUI app (Fyne/Go) + TUI (gocui fallback)

## Aesthetic Direction
- **Direction:** Industrial/Utilitarian
- **Decoration level:** Minimal — typography, color, and spacing do all the work; no ornamental elements
- **Mood:** A focused radio instrument. Not a 1990s FLTK app, not a toy. Feels like it belongs next to a modern Icom or Elecraft rig — purposeful, readable in low light, no wasted space.
- **Research basis:** The ham radio software landscape (fldigi, WSJT-X, CW Skimmer) is dominated by cluttered "functional retro" — pale yellow/blue, FLTK-era widgets. Modern outliers (HAMRS, HRD dark mode) lead with dark-mode minimalism. MoDe stands apart with a distinctive amber accent and near-black background.

## Typography
- **All roles:** Ubuntu Mono (Regular, Bold, Italic, Bold Italic)
  - Already embedded in the binary (`assets/fonts/UbuntuMono-*.ttf`)
  - Fixed-width is non-negotiable: callsigns, decoded text, frequencies, and the ASCII spectrogram all require monospace
  - Ubuntu Mono has slashed zeros — essential for distinguishing `0` from `O` in callsigns
- **Scale:**
  - App title / hero: 40px bold
  - Decoded text / body: 15px regular (current: `theme.SizeNameText * 1.5` — keep this)
  - Toolbar labels: 12px bold
  - Data readouts: 13px regular, tabular-nums
  - Spectrum / live indicator: 13px bold
  - Secondary labels / timestamps: 11px regular

## Color

- **Approach:** Restrained dark — one amber accent against a near-black background. Color is rare and meaningful.
- **Primary variant:** Dark mode (operators work in low-light shacks; dark screens are standard)

| Role | Hex | Usage |
|------|-----|-------|
| Background | `#0D1117` | App background, titlebar, statusbar |
| Surface | `#161B22` | Cards, input bg, toolbar, text area |
| Border | `#6E7681` | All borders (maps to `BorderedContainer`) |
| Foreground | `#E6EDF3` | Primary text — decoded Morse output |
| Muted | `#8B949E` | Toolbar labels, secondary info |
| **Accent** | `#E8890C` | Primary action, new/fresh decoded text, spectrum indicator, focus rings |
| Success | `#3FB950` | Signal acquired / RECEIVING status |
| Warning | `#D29922` | Marginal signal / ACQUIRING status |
| Error | `#F85149` | No signal / decode error |
| Hover | `#1F2937` | Button hover states |

**Dark mode is the primary and only supported variant.** Light mode support via Fyne's `ThemeVariant` is optional and not prioritized.

**Accent rationale:** Every Fyne app defaults to a blue `ColorNamePrimary`. Amber `#E8890C` reads as phosphor CRT — familiar and nostalgic for radio operators, readable in low light, and gives MoDe an instant visual identity that distinguishes it from every other Fyne app.

## Spacing

- **Base unit:** 4px
- **Density:** Compact (operators want maximum information density without scrolling)
- **Current `CompactTheme` padding:** 2px — intentionally tighter than Fyne default. Keep.
- **Scale:** 2(2px) 4(4px) 8(8px) 16(16px) 24(24px) 32(32px) 48(48px)

## Layout

- **Approach:** Grid-disciplined — strict horizontal toolbar, full-width text area, compact status bar
- **App structure:** Top toolbar | Center text area (flex fill) | Bottom status bar
- **Border radius:** 2px everywhere (`BorderedContainer` cornerRadius = 2). No bubbly rounding.
- **Toolbar:** Single horizontal row; all controls visible without menus or overflow
- **Text area:** Full width, word-wrap, vertical scroll only — the decoded text is the product

## Motion

- **Approach:** None — this is a real-time signal processing app. No entrance animations, no transitions beyond Fyne native widget redraws.
- **Exception:** The text cursor blink in the decoded output area can use a 1s step blink (`animation: blink 1s step-end infinite`) if implemented in a future web port.

## Fyne Theme Implementation Notes

The `CompactTheme` in `cmd/gmode/main.go` should override `Color()` to return the palette above using Fyne's `ThemeColorName` constants:

```go
func (t CompactTheme) Color(name fyne.ThemeColorName, variant fyne.ThemeVariant) color.Color {
    switch name {
    case theme.ColorNameBackground:
        return color.NRGBA{R: 0x0D, G: 0x11, B: 0x17, A: 0xFF} // #0D1117
    case theme.ColorNameForeground:
        return color.NRGBA{R: 0xE6, G: 0xED, B: 0xF3, A: 0xFF} // #E6EDF3
    case theme.ColorNamePrimary:
        return color.NRGBA{R: 0xE8, G: 0x89, B: 0x0C, A: 0xFF} // #E8890C amber
    case theme.ColorNameButton:
        return color.NRGBA{R: 0x2D, G: 0x33, B: 0x3B, A: 0xFF} // #2D333B
    case theme.ColorNameInputBackground:
        return color.NRGBA{R: 0x16, G: 0x1B, B: 0x22, A: 0xFF} // #161B22
    case theme.ColorNameDisabled:
        return color.NRGBA{R: 0x8B, G: 0x94, B: 0x9E, A: 0xFF} // #8B949E
    case theme.ColorNamePlaceHolder:
        return color.NRGBA{R: 0x8B, G: 0x94, B: 0x9E, A: 0xFF} // #8B949E
    case theme.ColorNameHover:
        return color.NRGBA{R: 0x1F, G: 0x29, B: 0x37, A: 0xFF} // #1F2937
    case theme.ColorNameFocus:
        return color.NRGBA{R: 0xE8, G: 0x89, B: 0x0C, A: 0x80} // amber, 50% alpha
    case theme.ColorNameSeparator:
        return color.NRGBA{R: 0x30, G: 0x36, B: 0x3D, A: 0xFF} // #30363D
    case theme.ColorNameScrollBar:
        return color.NRGBA{R: 0x30, G: 0x36, B: 0x3D, A: 0xFF} // #30363D
    case theme.ColorNameSuccess:
        return color.NRGBA{R: 0x3F, G: 0xB9, B: 0x50, A: 0xFF} // #3FB950
    case theme.ColorNameWarning:
        return color.NRGBA{R: 0xD2, G: 0x99, B: 0x22, A: 0xFF} // #D29922
    case theme.ColorNameError:
        return color.NRGBA{R: 0xF8, G: 0x51, B: 0x49, A: 0xFF} // #F85149
    }
    return theme.DefaultTheme().Color(name, variant)
}
```

The `BorderedContainer` border color is already derived from `theme.ColorNameForeground` — with the new theme, it will pick up `#E6EDF3` at 100% alpha, which is quite bright. Consider passing `theme.ColorNameSeparator` (`#30363D`) instead for a subtler border, matching the surface/border relationship.

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-19 | Initial design system created | Created by /design-consultation based on ham radio software research + MoDe codebase analysis |
| 2026-03-19 | Ubuntu Mono as sole typeface | Already embedded in binary; monospace is non-negotiable for CW text; slashed zeros essential for callsigns |
| 2026-03-19 | Amber `#E8890C` as primary accent | Phosphor CRT reference; distinguishes MoDe from every default-blue Fyne app; readable in low light |
| 2026-03-19 | Near-black `#0D1117` background | GitHub Dark baseline — proven comfortable for long sessions vs. harsh pure black |
| 2026-03-19 | 2px border radius everywhere | Keep existing `BorderedContainer` cornerRadius; no bubbly rounding for an instrument aesthetic |
