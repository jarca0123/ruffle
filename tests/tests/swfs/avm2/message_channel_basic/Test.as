package {
    import flash.display.Sprite;
    import flash.system.Worker;
    import flash.system.MessageChannel;

    public class Test extends Sprite {
        public function Test() {
            var w:Worker = Worker.current;
            var ch:MessageChannel = w.createMessageChannel(w);
            trace("available (empty) = " + ch.messageAvailable);
            ch.send(42);
            ch.send("hi");
            trace("available = " + ch.messageAvailable);
            trace("recv1 = " + ch.receive());
            trace("recv2 = " + ch.receive());
            trace("available (drained) = " + ch.messageAvailable);
            trace("recv empty = " + ch.receive());
        }
    }
}
