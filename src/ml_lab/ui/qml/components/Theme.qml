pragma Singleton
import QtQuick

QtObject {
    readonly property bool dark: appController.darkTheme
    readonly property color bg: dark ? "#0B1117" : "#F4F7F9"
    readonly property color sidebar: dark ? "#0E151C" : "#EAF0F4"
    readonly property color surface: dark ? "#111820" : "#FFFFFF"
    readonly property color surfaceAlt: dark ? "#131C25" : "#F7FAFC"
    readonly property color hover: dark ? "#18232D" : "#EAF2F6"
    readonly property color border: dark ? "#25313D" : "#D5DEE6"
    readonly property color borderStrong: dark ? "#31404F" : "#B9C7D2"
    readonly property color text: dark ? "#F0F5F9" : "#18232D"
    readonly property color text2: dark ? "#E5EDF4" : "#30404D"
    readonly property color muted: dark ? "#97A8B8" : "#647889"
    readonly property color dim: dark ? "#708292" : "#7A8A97"
    readonly property color accent: dark ? "#86E4D0" : "#0A8F78"
    readonly property color accentHover: dark ? "#9EEAD9" : "#087966"
    readonly property color accentText: dark ? "#071018" : "#FFFFFF"
    readonly property color accentSurface: dark ? "#19302E" : "#DDF5F0"
    readonly property color accentBorder: dark ? "#2B5C53" : "#98D8CA"
    readonly property color goodSurface: dark ? "#15342E" : "#E0F5ED"
    readonly property color goodBorder: dark ? "#2D6D5D" : "#8DCEB8"
    readonly property color goodText: dark ? "#9EEAD9" : "#176B57"
    readonly property color badSurface: dark ? "#3A2025" : "#FBE7EA"
    readonly property color badBorder: dark ? "#7A3A45" : "#E2A1AA"
    readonly property color badText: dark ? "#F2A6B1" : "#9D3142"
    readonly property color warnSurface: dark ? "#3A301B" : "#FFF4D4"
    readonly property color warnBorder: dark ? "#806923" : "#D7B654"
    readonly property color warnText: dark ? "#EBCF71" : "#7A5C00"
}
