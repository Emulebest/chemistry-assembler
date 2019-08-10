module Main where

import Lib
import Network.Transport.TCP (createTransport, defaultTCPParameters)
import Control.Distributed.Process
import Control.Distributed.Process.Node
import Control.Concurrent (threadDelay)

hi msg = putStrLn "hello"

main :: IO ()
main = do
     Right t <- createTransport "127.0.0.1" "10501" ((,) "127.0.0.1") defaultTCPParameters
     node <- newLocalNode t initRemoteTable
     _ <- runProcess node $ do
       -- get our own process id
       self <- getSelfPid
       send self "hello"
       say "hi"
       hello <- expect :: Process String
       liftIO $ putStrLn hello
     liftIO $ threadDelay 200000
     return ()

