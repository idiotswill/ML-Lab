import QtQuick
import "."

Rectangle {
    id: root
    property string text: ""
    property string tone: "neutral"
    implicitWidth: label.implicitWidth + 18
    implicitHeight: 25
    radius: 12
    color: tone === "good" ? Theme.goodSurface : tone === "bad" ? Theme.badSurface : tone === "warn" ? Theme.warnSurface : Theme.surfaceAlt
    border.color: tone === "good" ? Theme.goodBorder : tone === "bad" ? Theme.badBorder : tone === "warn" ? Theme.warnBorder : Theme.borderStrong
    Text {
        id: label
        anchors.centerIn: parent
        text: root.text
        color: root.tone === "good" ? Theme.goodText : root.tone === "bad" ? Theme.badText : root.tone === "warn" ? Theme.warnText : Theme.muted
        font.pixelSize: 11
        font.weight: Font.DemiBold
    }
}
