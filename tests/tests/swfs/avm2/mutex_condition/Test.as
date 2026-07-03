package {
    import flash.display.Sprite;
    import flash.concurrent.Mutex;
    import flash.concurrent.Condition;

    public class Test extends Sprite {
        public function Test() {
            var m:Mutex = new Mutex();
            trace("mutex isSupported = " + Mutex.isSupported);
            m.lock();
            trace("tryLock (recursive) = " + m.tryLock());
            m.unlock();
            m.unlock();
            trace("tryLock after release = " + m.tryLock());
            m.unlock();

            var c:Condition = new Condition(m);
            trace("condition isSupported = " + Condition.isSupported);
            trace("ok");
        }
    }
}
