import os
import sys
sys.path.append("../")
from random import randint,shuffle
import threading
from time import sleep
import gc

try:
    import queue
except ImportError:
    import Queue as queue

class DataLoader(object):
    def __init__(self,img_list,
                 batch_size=32,
                 num_product=20,
                 prefetch_size=5,
                 min_prefetch_th=0,
                 num_worker=4,
                 handler_obj=None):


        self.batch_size=batch_size
        self.num_product=num_product
        self.prefetch_size=prefetch_size
        self.min_prefetch_th=min_prefetch_th
        self.num_worker=num_worker

        self.handler_obj=handler_obj

        self.img_list = img_list

        return


    def get_data_batch(self):
        pass
        while self.producer_queue.empty():
            if self.batch_finished and self.producer_queue.empty():
                return None
            continue

        cur_batch = self.producer_queue.get()
        if self.producer_queue.qsize() < self.min_prefetch_th:
            print "Prefetching:",self.batch_finished,self.producer_queue.qsize(),self.prefetch_size

            while ((not self.batch_finished) and self.producer_queue.qsize() <= self.prefetch_size):
                continue

        print "Query size: ", self.producer_queue.qsize()
        return cur_batch

    def producer_worker(self, worker_id):
        pass
        print "Producer worker started, worker id:", worker_id, "..."
        while True:
            if self.batch_finished:
                break

            is_all_finished=False
            batch_id_list = []
            with self.im_pos_locker:

                for i in xrange(0,self.batch_size):
                    if self.im_pos >= len(self.img_list):
                        is_all_finished = True
                        break
                    img_id = self.img_list[self.im_pos]
                    batch_id_list.append(img_id)
                    self.im_pos += 1

            if len(batch_id_list)==0:
                print "Exit by batch id list==0"
                self.batch_finished = True
                break

            ret = self.handler_obj.get_internal_data_batch(batch_id_list)

            if ret is None:
                print "Exit by ret is None"
                self.batch_finished = True
                break
            else:
                #   have data batch
                pass

                while self.producer_queue.full():
                    sleep(0.01)
                    continue

                self.producer_queue.put(ret)

            if is_all_finished:
                print "Exited by is all finished..."
                self.batch_finished=True
                break

        print "Finished worker id: ", worker_id, "..."


    def shuffle_data(self):
        pass
        shuffle(self.img_list)

    def reset_reader(self):
        pass
        try:
            self.end_producer_queue()
            self.producer_queue.clear()
        except:
            pass

        #   reset the data pos to position 0
        self.im_pos = 0
        self.start_producer_queue()


    def start_producer_queue(self):

        self.batch_finished = False
        self.producer_queue = queue.Queue(self.num_product)
        self.im_pos_locker = threading.Lock()
        self.worker_th_list = []

        for i in xrange(0, self.num_worker):
            worker_th = threading.Thread(target=self.producer_worker, args=(i,))
            worker_th.start()
            self.worker_th_list.append(worker_th)

        print "Prefecting [Start Producer Queue]..."
        while self.producer_queue.qsize() <= self.prefetch_size and not self.batch_finished:
            continue

        return

    def end_producer_queue(self):

        self.batch_finished = True
        self.worker_th_list = []

        return

