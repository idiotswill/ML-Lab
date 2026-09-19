import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Item {
    id: root

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 2

                Text {
                    text: "Models & Registry"
                    color: Theme.text
                    font.pixelSize: 24
                    font.weight: Font.DemiBold
                }

                Text {
                    text: "Immutable model records · compatibility · sequential promotion history"
                    color: Theme.muted
                    font.pixelSize: 12
                }
            }

            StatusPill {
                text: "INTEGRATION NO-GO"
                tone: "warn"
            }
        }

        Rectangle {
            Layout.fillWidth: true
            implicitHeight: registryMessage.implicitHeight + 20
            radius: 8
            color: Theme.surfaceAlt
            border.color: Theme.border

            Text {
                id: registryMessage
                anchors.fill: parent
                anchors.margins: 10
                text: appController.modelsRegistry.promotionMessage
                color: Theme.muted
                font.pixelSize: 12
                wrapMode: Text.WordWrap
                verticalAlignment: Text.AlignVCenter
            }
        }

        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal

            Panel {
                SplitView.preferredWidth: 280
                SplitView.minimumWidth: 160

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true
                        Text {
                            Layout.fillWidth: true
                            text: "READY TO REGISTER"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                        }
                        Text {
                            text: appController.modelsRegistry.registerableExperiments.length
                            color: Theme.muted
                            font.pixelSize: 11
                        }
                    }

                    ListView {
                        id: registerableList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 6
                        model: appController.modelsRegistry.registerableExperiments

                        delegate: Rectangle {
                            required property var modelData
                            width: registerableList.width
                            height: 104
                            radius: 7
                            color: Theme.surfaceAlt
                            border.color: Theme.border

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 9
                                spacing: 3

                                Text {
                                    text: "Experiment " + modelData.shortId
                                    color: Theme.text
                                    font.pixelSize: 12
                                    font.weight: Font.DemiBold
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.trainer + " · " + modelData.runtime
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    elide: Text.ElideRight
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.modelSha256
                                    color: Theme.dim
                                    font.pixelSize: 9
                                    elide: Text.ElideMiddle
                                }
                                Item { Layout.fillHeight: true }
                                LabButton {
                                    text: "Register model"
                                    primary: true
                                    Layout.fillWidth: true
                                    onClicked: appController.modelsRegistry.registerExperiment(modelData.id)
                                }
                            }
                        }

                        Text {
                            anchors.centerIn: parent
                            visible: registerableList.count === 0
                            text: "No unregistered completed models."
                            color: Theme.dim
                            font.pixelSize: 12
                        }
                    }
                }
            }

            Panel {
                SplitView.preferredWidth: 310
                SplitView.minimumWidth: 170

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true
                        Text {
                            Layout.fillWidth: true
                            text: "REGISTERED MODELS"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                        }
                        Text {
                            text: appController.modelsRegistry.modelTotal + " total"
                            color: Theme.muted
                            font.pixelSize: 11
                        }
                    }

                    ComboBox {
                        Layout.fillWidth: true
                        model: appController.modelsRegistry.modelStageFilters
                        currentIndex: Math.max(
                            0,
                            appController.modelsRegistry.modelStageFilters.indexOf(
                                appController.modelsRegistry.modelStageFilter
                            )
                        )
                        Accessible.name: "Model stage filter"
                        onActivated: appController.modelsRegistry.setModelStageFilter(currentText)
                    }

                    ListView {
                        id: modelList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 5
                        model: appController.modelsRegistry.modelRows

                        delegate: ItemDelegate {
                            required property var modelData
                            width: modelList.width
                            height: 78
                            activeFocusOnTab: true
                            Accessible.name: "Model " + modelData.shortId + " " + modelData.stage
                            onClicked: appController.modelsRegistry.selectModel(modelData.id)

                            contentItem: ColumnLayout {
                                spacing: 3
                                RowLayout {
                                    Layout.fillWidth: true
                                    Text {
                                        Layout.fillWidth: true
                                        text: modelData.shortId + " · exp " + modelData.experimentShortId
                                        color: Theme.text
                                        font.pixelSize: 12
                                        font.weight: Font.DemiBold
                                        elide: Text.ElideRight
                                    }
                                    StatusPill {
                                        text: modelData.stage
                                        tone: modelData.stage === "RELEASE_CANDIDATE" ? "good" : "neutral"
                                    }
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.modelSha256
                                    color: Theme.dim
                                    font.pixelSize: 9
                                    elide: Text.ElideMiddle
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.updatedAt
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    elide: Text.ElideRight
                                }
                            }

                            background: Rectangle {
                                radius: 7
                                color: appController.modelsRegistry.selectedModelId === modelData.id ?
                                       Theme.accentSurface : (parent.hovered ? Theme.hover : "transparent")
                                border.color: parent.activeFocus ? Theme.accent :
                                              (appController.modelsRegistry.selectedModelId === modelData.id ?
                                               Theme.accentBorder : "transparent")
                                border.width: parent.activeFocus ? 2 : 1
                            }
                        }

                        Text {
                            anchors.centerIn: parent
                            visible: modelList.count === 0
                            text: "No registered models on this page."
                            color: Theme.dim
                            font.pixelSize: 12
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 6

                        Text {
                            Layout.fillWidth: true
                            text: "Page " + appController.modelsRegistry.modelPageNumber
                            color: Theme.muted
                            font.pixelSize: 10
                        }

                        LabButton {
                            text: "Previous"
                            enabled: appController.modelsRegistry.canPreviousModelPage
                            onClicked: appController.modelsRegistry.previousModelPage()
                        }

                        LabButton {
                            text: "Next"
                            enabled: appController.modelsRegistry.canNextModelPage
                            onClicked: appController.modelsRegistry.nextModelPage()
                        }
                    }
                }
            }

            Panel {
                SplitView.fillWidth: true
                SplitView.minimumWidth: 220

                ScrollView {
                    id: registryDetailsScroll
                    anchors.fill: parent
                    clip: true
                    ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                    ColumnLayout {
                        width: Math.max(registryDetailsScroll.availableWidth - 20, 1)
                        spacing: 10

                        Text {
                            text: appController.modelsRegistry.selectedModel.id ?
                                  "Model " + appController.modelsRegistry.selectedModel.shortId :
                                  "Model details"
                            color: Theme.text
                            font.pixelSize: 18
                            font.weight: Font.DemiBold
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            visible: !!appController.modelsRegistry.selectedModel.id

                            StatusPill {
                                text: appController.modelsRegistry.selectedModel.stage || ""
                                tone: appController.modelsRegistry.selectedModel.stage === "RELEASE_CANDIDATE" ?
                                      "good" : "neutral"
                            }
                            Text {
                                Layout.fillWidth: true
                                text: "Experiment " +
                                      (appController.modelsRegistry.selectedModel.experimentShortId || "")
                                color: Theme.muted
                                font.pixelSize: 11
                            }
                        }

                        Text {
                            text: "MODEL SHA-256"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                            visible: !!appController.modelsRegistry.selectedModel.id
                        }
                        Text {
                            Layout.fillWidth: true
                            text: appController.modelsRegistry.selectedModel.modelSha256 || "—"
                            color: Theme.text2
                            font.pixelSize: 10
                            wrapMode: Text.WrapAnywhere
                            visible: !!appController.modelsRegistry.selectedModel.id
                        }

                        Text {
                            text: "MANIFEST SHA-256"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                            visible: !!appController.modelsRegistry.selectedModel.id
                        }
                        Text {
                            Layout.fillWidth: true
                            text: appController.modelsRegistry.selectedModel.manifestSha256 || "—"
                            color: Theme.text2
                            font.pixelSize: 10
                            wrapMode: Text.WrapAnywhere
                            visible: !!appController.modelsRegistry.selectedModel.id
                        }

                        Text {
                            text: "COMPATIBILITY"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                            visible: !!appController.modelsRegistry.selectedModel.id
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            implicitHeight: compatibilityText.implicitHeight + 18
                            radius: 6
                            color: Theme.surfaceAlt
                            border.color: Theme.border
                            visible: !!appController.modelsRegistry.selectedModel.id

                            Text {
                                id: compatibilityText
                                anchors.fill: parent
                                anchors.margins: 9
                                text: appController.modelsRegistry.selectedModel.compatibility || "{}"
                                color: Theme.text2
                                font.pixelSize: 10
                                font.family: "Consolas"
                                wrapMode: Text.WrapAnywhere
                            }
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            height: 1
                            color: Theme.border
                            visible: !!appController.modelsRegistry.selectedModel.id
                        }

                        Text {
                            text: "PROMOTION"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                            visible: !!appController.modelsRegistry.selectedModel.id
                        }

                        TextField {
                            id: promotionNote
                            Layout.fillWidth: true
                            placeholderText: "Evidence note (optional)"
                            enabled: appController.modelsRegistry.canPromote
                            visible: !!appController.modelsRegistry.selectedModel.id
                        }

                        LabButton {
                            Layout.fillWidth: true
                            text: appController.modelsRegistry.nextStage ?
                                  "Promote to " + appController.modelsRegistry.nextStage :
                                  "No ordinary promotion available"
                            primary: true
                            enabled: appController.modelsRegistry.canPromote
                            visible: !!appController.modelsRegistry.selectedModel.id
                            onClicked: {
                                appController.modelsRegistry.promoteSelected(promotionNote.text)
                                promotionNote.clear()
                            }
                        }

                        Text {
                            text: "STAGE HISTORY"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                            visible: !!appController.modelsRegistry.selectedModel.id
                        }

                        Repeater {
                            model: appController.modelsRegistry.stageHistory

                            delegate: Rectangle {
                                required property var modelData
                                Layout.fillWidth: true
                                implicitHeight: historyColumn.implicitHeight + 16
                                radius: 6
                                color: Theme.surfaceAlt
                                border.color: Theme.border

                                ColumnLayout {
                                    id: historyColumn
                                    anchors.fill: parent
                                    anchors.margins: 8
                                    spacing: 3

                                    Text {
                                        text: modelData.fromStage + " → " + modelData.toStage
                                        color: Theme.text
                                        font.pixelSize: 11
                                        font.weight: Font.DemiBold
                                    }
                                    Text {
                                        text: modelData.createdAt
                                        color: Theme.dim
                                        font.pixelSize: 9
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: modelData.evidence
                                        color: Theme.muted
                                        font.pixelSize: 9
                                        font.family: "Consolas"
                                        wrapMode: Text.WrapAnywhere
                                    }
                                }
                            }
                        }

                        Text {
                            Layout.fillWidth: true
                            visible: !appController.modelsRegistry.selectedModel.id
                            text: "Select a registered model to inspect compatibility and promotion history."
                            color: Theme.dim
                            font.pixelSize: 12
                            wrapMode: Text.WordWrap
                        }
                    }
                }
            }
        }
    }
}
