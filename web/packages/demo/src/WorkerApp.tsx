import { useState, useCallback, DragEvent } from "react";
import type { Config } from "ruffle-core";
import { Navbar } from "./navbar.tsx";
import { MetadataPanel } from "./metadata.tsx";
import { WorkerPlayerView } from "./WorkerPlayerView.tsx";

interface WorkerAppProperties {
    allowUrlLoading: boolean;
    allowSampleSwfs: boolean;
}

/**
 * The same layout/chrome as the demo's {@link App}, but the movie runs on a
 * worker (see {@link WorkerPlayerView}). Selecting a sample/URL/file restarts the
 * worker player with it; metadata isn't reported by the worker player yet, so the
 * panel stays empty.
 */
export function WorkerApp({
    allowUrlLoading,
    allowSampleSwfs,
}: WorkerAppProperties) {
    const [selectedUrl, setSelectedUrl] = useState<string>("/openttd.swf");
    const [reloadNonce, setReloadNonce] = useState<number>(0);
    const [metadataVisible, setMetadataVisible] = useState<boolean>(false);
    const [selectedFilename, setSelectedFilename] = useState<string | null>(
        null,
    );
    const [dragOverlayVisible, setDragOverlayVisible] =
        useState<boolean>(false);

    const toggleMetadataVisible = () => setMetadataVisible((v) => !v);

    // A fresh key remounts the player view onto a new canvas (needed because a
    // canvas can only be transferred to a worker once).
    const reloadMovie = () => setReloadNonce((n) => n + 1);

    const onSelectUrl = useCallback(
        (url: string, _options: Config.BaseLoadOptions) => {
            setSelectedUrl(url);
        },
        [],
    );

    const onSelectFile = (file: File) => {
        setSelectedFilename(file.name);
        setSelectedUrl(URL.createObjectURL(file));
    };

    const onFileDragEnter = (event: DragEvent<HTMLElement>) => {
        event.stopPropagation();
        event.preventDefault();
    };

    const onFileDragLeave = (event: DragEvent<HTMLElement>) => {
        event.stopPropagation();
        event.preventDefault();
        setDragOverlayVisible(false);
    };

    const onFileDragOver = (event: DragEvent<HTMLElement>) => {
        event.stopPropagation();
        event.preventDefault();
        setDragOverlayVisible(true);
    };

    const onFileDragDrop = (event: DragEvent<HTMLElement>) => {
        event.stopPropagation();
        event.preventDefault();
        setDragOverlayVisible(false);
        if (event.dataTransfer && event.dataTransfer.files.length > 0) {
            onSelectFile(event.dataTransfer.files[0]!);
        }
    };

    return (
        <>
            <Navbar
                allowUrlLoading={allowUrlLoading}
                allowSampleSwfs={allowSampleSwfs}
                onToggleMetadata={toggleMetadataVisible}
                onReloadMovie={reloadMovie}
                onSelectUrl={onSelectUrl}
                onSelectFile={onSelectFile}
                selectedFilename={selectedFilename}
                setSelectedFilename={setSelectedFilename}
                onFileDragLeave={onFileDragLeave}
                onFileDragOver={onFileDragOver}
                onFileDragDrop={onFileDragDrop}
            />

            <div
                id="main"
                className={metadataVisible ? "info-container-shown" : ""}
            >
                <div
                    id="player-container"
                    aria-label="Select a demo or drag an SWF"
                    onDragEnter={onFileDragEnter}
                    onDragLeave={onFileDragLeave}
                    onDragOver={onFileDragOver}
                    onDrop={onFileDragDrop}
                >
                    <WorkerPlayerView
                        // Key + pass the *resolved absolute* URL: the Navbar
                        // auto-selects the sample on mount (`openttd.swf`), which
                        // must not differ from the default (`/openttd.swf`) — else
                        // the view remounts and churns a second worker player.
                        key={`${new URL(selectedUrl, location.href).href}#${reloadNonce}`}
                        swfUrl={new URL(selectedUrl, location.href).href}
                    />
                    <div
                        id="overlay"
                        className={dragOverlayVisible ? "drag" : ""}
                    ></div>
                </div>
                <MetadataPanel visible={metadataVisible} metadata={null} />
            </div>
        </>
    );
}
