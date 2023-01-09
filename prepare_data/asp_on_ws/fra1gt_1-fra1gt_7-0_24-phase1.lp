
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(fra1gt_1,102,0). coords(fra1gt_1,277,35,0).  happensAt(appear(fra1gt_1),0). happensAt(walking(fra1gt_1),0). orientation(fra1gt_7,136,0). coords(fra1gt_7,141,45,0).  happensAt(appear(fra1gt_7),0). happensAt(walking(fra1gt_7),0).
orientation(fra1gt_1,102,1). coords(fra1gt_1,277,34,1).  visible(fra1gt_1,1). happensAt(walking(fra1gt_1),1). orientation(fra1gt_7,136,1). coords(fra1gt_7,142,45,1).  visible(fra1gt_7,1). happensAt(walking(fra1gt_7),1).
orientation(fra1gt_1,102,2). coords(fra1gt_1,277,34,2).  visible(fra1gt_1,2). happensAt(walking(fra1gt_1),2). orientation(fra1gt_7,136,2). coords(fra1gt_7,141,44,2).  visible(fra1gt_7,2). happensAt(walking(fra1gt_7),2).
orientation(fra1gt_1,102,3). coords(fra1gt_1,277,33,3).  visible(fra1gt_1,3). happensAt(walking(fra1gt_1),3). orientation(fra1gt_7,136,3). coords(fra1gt_7,141,43,3).  visible(fra1gt_7,3). happensAt(walking(fra1gt_7),3).
orientation(fra1gt_1,102,4). coords(fra1gt_1,277,31,4).  visible(fra1gt_1,4). happensAt(walking(fra1gt_1),4). orientation(fra1gt_7,136,4). coords(fra1gt_7,141,43,4).  visible(fra1gt_7,4). happensAt(walking(fra1gt_7),4).
orientation(fra1gt_1,102,5). coords(fra1gt_1,277,30,5).  visible(fra1gt_1,5). happensAt(walking(fra1gt_1),5). orientation(fra1gt_7,136,5). coords(fra1gt_7,141,43,5).  visible(fra1gt_7,5). happensAt(walking(fra1gt_7),5).
orientation(fra1gt_1,112,6). coords(fra1gt_1,277,28,6).  visible(fra1gt_1,6). happensAt(walking(fra1gt_1),6). orientation(fra1gt_7,136,6). coords(fra1gt_7,140,43,6).  visible(fra1gt_7,6). happensAt(walking(fra1gt_7),6).
orientation(fra1gt_1,112,7). coords(fra1gt_1,277,26,7).  visible(fra1gt_1,7). happensAt(walking(fra1gt_1),7). orientation(fra1gt_7,136,7). coords(fra1gt_7,140,43,7).  visible(fra1gt_7,7). happensAt(walking(fra1gt_7),7).
orientation(fra1gt_1,112,8). coords(fra1gt_1,277,24,8).  visible(fra1gt_1,8). happensAt(walking(fra1gt_1),8). orientation(fra1gt_7,136,8). coords(fra1gt_7,141,43,8).  visible(fra1gt_7,8). happensAt(walking(fra1gt_7),8).
orientation(fra1gt_1,112,9). coords(fra1gt_1,277,23,9).  visible(fra1gt_1,9). happensAt(walking(fra1gt_1),9). orientation(fra1gt_7,136,9). coords(fra1gt_7,141,43,9).  visible(fra1gt_7,9). happensAt(walking(fra1gt_7),9).
orientation(fra1gt_1,112,10). coords(fra1gt_1,277,21,10).  visible(fra1gt_1,10). happensAt(walking(fra1gt_1),10). orientation(fra1gt_7,136,10). coords(fra1gt_7,141,42,10).  visible(fra1gt_7,10). happensAt(walking(fra1gt_7),10).
orientation(fra1gt_1,112,11). coords(fra1gt_1,277,21,11).  visible(fra1gt_1,11). happensAt(walking(fra1gt_1),11). orientation(fra1gt_7,136,11). coords(fra1gt_7,141,42,11).  visible(fra1gt_7,11). happensAt(walking(fra1gt_7),11).
orientation(fra1gt_1,112,12). coords(fra1gt_1,277,20,12).  visible(fra1gt_1,12). happensAt(walking(fra1gt_1),12). orientation(fra1gt_7,136,12). coords(fra1gt_7,141,42,12).  visible(fra1gt_7,12). happensAt(walking(fra1gt_7),12).
orientation(fra1gt_1,112,13). coords(fra1gt_1,277,20,13).  visible(fra1gt_1,13). happensAt(walking(fra1gt_1),13). orientation(fra1gt_7,136,13). coords(fra1gt_7,141,42,13).  visible(fra1gt_7,13). happensAt(walking(fra1gt_7),13).
orientation(fra1gt_1,112,14). coords(fra1gt_1,277,20,14).  visible(fra1gt_1,14). happensAt(walking(fra1gt_1),14). orientation(fra1gt_7,136,14). coords(fra1gt_7,141,42,14).  visible(fra1gt_7,14). happensAt(walking(fra1gt_7),14).
orientation(fra1gt_1,112,15). coords(fra1gt_1,277,20,15).  visible(fra1gt_1,15). happensAt(walking(fra1gt_1),15). orientation(fra1gt_7,136,15). coords(fra1gt_7,141,42,15).  visible(fra1gt_7,15). happensAt(walking(fra1gt_7),15).
orientation(fra1gt_1,112,16). coords(fra1gt_1,277,20,16).  visible(fra1gt_1,16). happensAt(walking(fra1gt_1),16). orientation(fra1gt_7,136,16). coords(fra1gt_7,141,42,16).  visible(fra1gt_7,16). happensAt(walking(fra1gt_7),16).
orientation(fra1gt_1,112,17). coords(fra1gt_1,277,20,17).  visible(fra1gt_1,17). happensAt(walking(fra1gt_1),17). orientation(fra1gt_7,136,17). coords(fra1gt_7,142,42,17).  visible(fra1gt_7,17). happensAt(walking(fra1gt_7),17).
orientation(fra1gt_1,112,18). coords(fra1gt_1,277,19,18).  visible(fra1gt_1,18). happensAt(walking(fra1gt_1),18). orientation(fra1gt_7,136,18). coords(fra1gt_7,142,42,18).  visible(fra1gt_7,18). happensAt(walking(fra1gt_7),18).
orientation(fra1gt_1,112,19). coords(fra1gt_1,277,19,19).  visible(fra1gt_1,19). happensAt(walking(fra1gt_1),19). orientation(fra1gt_7,136,19). coords(fra1gt_7,142,42,19).  visible(fra1gt_7,19). happensAt(walking(fra1gt_7),19).
orientation(fra1gt_1,112,20). coords(fra1gt_1,277,18,20).  visible(fra1gt_1,20). happensAt(walking(fra1gt_1),20). orientation(fra1gt_7,136,20). coords(fra1gt_7,142,42,20).  visible(fra1gt_7,20). happensAt(walking(fra1gt_7),20).
orientation(fra1gt_1,112,21). coords(fra1gt_1,277,16,21).  visible(fra1gt_1,21). happensAt(walking(fra1gt_1),21). orientation(fra1gt_7,136,21). coords(fra1gt_7,142,42,21).  visible(fra1gt_7,21). happensAt(walking(fra1gt_7),21).
orientation(fra1gt_1,108,22). coords(fra1gt_1,277,16,22).  visible(fra1gt_1,22). happensAt(walking(fra1gt_1),22). orientation(fra1gt_7,136,22). coords(fra1gt_7,142,42,22).  visible(fra1gt_7,22). happensAt(walking(fra1gt_7),22).
orientation(fra1gt_1,106,23). coords(fra1gt_1,277,14,23).  visible(fra1gt_1,23). happensAt(walking(fra1gt_1),23). orientation(fra1gt_7,136,23). coords(fra1gt_7,143,41,23).  visible(fra1gt_7,23). happensAt(walking(fra1gt_7),23).
orientation(fra1gt_1,106,24). coords(fra1gt_1,278,12,24).  visible(fra1gt_1,24). happensAt(walking(fra1gt_1),24). orientation(fra1gt_7,136,24). coords(fra1gt_7,144,40,24).  visible(fra1gt_7,24). happensAt(walking(fra1gt_7),24).
#show holdsAt/2.