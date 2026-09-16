import QtQuick
import QtQuick.Controls
import "."

Button {
    id: control
    property bool primary: false
    implicitHeight: 38
    leftPadding: 16
    rightPadding: 16
    font.pixelSize: 13
    font.weight: Font.DemiBold
    contentItem: Text {
        text: control.text
        color: control.enabled ? (control.primary ? Theme.accentText : Theme.text2) : Theme.dim
        font: control.font
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
    background: Rectangle {
        radius: 8
        color: !control.enabled ? Theme.surfaceAlt : control.down ?
               (control.primary ? Theme.accentHover : Theme.hover) :
               control.hovered ? (control.primary ? Theme.accentHover : Theme.hover) :
               (control.primary ? Theme.accent : Theme.surfaceAlt)
        border.color: control.primary ? Theme.accent : Theme.borderStrong
        border.width: 1
    }
}
