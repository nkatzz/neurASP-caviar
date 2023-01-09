
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

	orientation(fra1gt_0,145,0). coords(fra1gt_0,190,143,0).  happensAt(appear(fra1gt_0),0). happensAt(inactive(fra1gt_0),0). orientation(fra1gt_1,102,0). coords(fra1gt_1,277,35,0).  happensAt(appear(fra1gt_1),0). happensAt(walking(fra1gt_1),0).
orientation(fra1gt_0,145,1). coords(fra1gt_0,190,143,1).  visible(fra1gt_0,1). happensAt(inactive(fra1gt_0),1). orientation(fra1gt_1,102,1). coords(fra1gt_1,277,34,1).  visible(fra1gt_1,1). happensAt(walking(fra1gt_1),1).
orientation(fra1gt_0,145,2). coords(fra1gt_0,190,143,2).  visible(fra1gt_0,2). happensAt(inactive(fra1gt_0),2). orientation(fra1gt_1,102,2). coords(fra1gt_1,277,34,2).  visible(fra1gt_1,2). happensAt(walking(fra1gt_1),2).
orientation(fra1gt_0,145,3). coords(fra1gt_0,190,143,3).  visible(fra1gt_0,3). happensAt(inactive(fra1gt_0),3). orientation(fra1gt_1,102,3). coords(fra1gt_1,277,33,3).  visible(fra1gt_1,3). happensAt(walking(fra1gt_1),3).
orientation(fra1gt_0,145,4). coords(fra1gt_0,190,143,4).  visible(fra1gt_0,4). happensAt(inactive(fra1gt_0),4). orientation(fra1gt_1,102,4). coords(fra1gt_1,277,31,4).  visible(fra1gt_1,4). happensAt(walking(fra1gt_1),4).
orientation(fra1gt_0,145,5). coords(fra1gt_0,190,143,5).  visible(fra1gt_0,5). happensAt(inactive(fra1gt_0),5). orientation(fra1gt_1,102,5). coords(fra1gt_1,277,30,5).  visible(fra1gt_1,5). happensAt(walking(fra1gt_1),5).
orientation(fra1gt_0,145,6). coords(fra1gt_0,190,143,6).  visible(fra1gt_0,6). happensAt(inactive(fra1gt_0),6). orientation(fra1gt_1,112,6). coords(fra1gt_1,277,28,6).  visible(fra1gt_1,6). happensAt(walking(fra1gt_1),6).
orientation(fra1gt_0,145,7). coords(fra1gt_0,190,143,7).  visible(fra1gt_0,7). happensAt(inactive(fra1gt_0),7). orientation(fra1gt_1,112,7). coords(fra1gt_1,277,26,7).  visible(fra1gt_1,7). happensAt(walking(fra1gt_1),7).
orientation(fra1gt_0,145,8). coords(fra1gt_0,190,143,8).  visible(fra1gt_0,8). happensAt(inactive(fra1gt_0),8). orientation(fra1gt_1,112,8). coords(fra1gt_1,277,24,8).  visible(fra1gt_1,8). happensAt(walking(fra1gt_1),8).
orientation(fra1gt_0,145,9). coords(fra1gt_0,190,143,9).  visible(fra1gt_0,9). happensAt(inactive(fra1gt_0),9). orientation(fra1gt_1,112,9). coords(fra1gt_1,277,23,9).  visible(fra1gt_1,9). happensAt(walking(fra1gt_1),9).
orientation(fra1gt_0,145,10). coords(fra1gt_0,190,143,10).  visible(fra1gt_0,10). happensAt(inactive(fra1gt_0),10). orientation(fra1gt_1,112,10). coords(fra1gt_1,277,21,10).  visible(fra1gt_1,10). happensAt(walking(fra1gt_1),10).
orientation(fra1gt_0,145,11). coords(fra1gt_0,190,143,11).  visible(fra1gt_0,11). happensAt(inactive(fra1gt_0),11). orientation(fra1gt_1,112,11). coords(fra1gt_1,277,21,11).  visible(fra1gt_1,11). happensAt(walking(fra1gt_1),11).
orientation(fra1gt_0,145,12). coords(fra1gt_0,190,143,12).  visible(fra1gt_0,12). happensAt(inactive(fra1gt_0),12). orientation(fra1gt_1,112,12). coords(fra1gt_1,277,20,12).  visible(fra1gt_1,12). happensAt(walking(fra1gt_1),12).
orientation(fra1gt_0,145,13). coords(fra1gt_0,190,143,13).  visible(fra1gt_0,13). happensAt(inactive(fra1gt_0),13). orientation(fra1gt_1,112,13). coords(fra1gt_1,277,20,13).  visible(fra1gt_1,13). happensAt(walking(fra1gt_1),13).
orientation(fra1gt_0,145,14). coords(fra1gt_0,191,143,14).  visible(fra1gt_0,14). happensAt(inactive(fra1gt_0),14). orientation(fra1gt_1,112,14). coords(fra1gt_1,277,20,14).  visible(fra1gt_1,14). happensAt(walking(fra1gt_1),14).
orientation(fra1gt_0,145,15). coords(fra1gt_0,191,143,15).  visible(fra1gt_0,15). happensAt(inactive(fra1gt_0),15). orientation(fra1gt_1,112,15). coords(fra1gt_1,277,20,15).  visible(fra1gt_1,15). happensAt(walking(fra1gt_1),15).
orientation(fra1gt_0,145,16). coords(fra1gt_0,191,143,16).  visible(fra1gt_0,16). happensAt(inactive(fra1gt_0),16). orientation(fra1gt_1,112,16). coords(fra1gt_1,277,20,16).  visible(fra1gt_1,16). happensAt(walking(fra1gt_1),16).
orientation(fra1gt_0,145,17). coords(fra1gt_0,191,143,17).  visible(fra1gt_0,17). happensAt(inactive(fra1gt_0),17). orientation(fra1gt_1,112,17). coords(fra1gt_1,277,20,17).  visible(fra1gt_1,17). happensAt(walking(fra1gt_1),17).
orientation(fra1gt_0,145,18). coords(fra1gt_0,191,144,18).  visible(fra1gt_0,18). happensAt(inactive(fra1gt_0),18). orientation(fra1gt_1,112,18). coords(fra1gt_1,277,19,18).  visible(fra1gt_1,18). happensAt(walking(fra1gt_1),18).
orientation(fra1gt_0,145,19). coords(fra1gt_0,191,144,19).  visible(fra1gt_0,19). happensAt(inactive(fra1gt_0),19). orientation(fra1gt_1,112,19). coords(fra1gt_1,277,19,19).  visible(fra1gt_1,19). happensAt(walking(fra1gt_1),19).
orientation(fra1gt_0,145,20). coords(fra1gt_0,191,144,20).  visible(fra1gt_0,20). happensAt(inactive(fra1gt_0),20). orientation(fra1gt_1,112,20). coords(fra1gt_1,277,18,20).  visible(fra1gt_1,20). happensAt(walking(fra1gt_1),20).
orientation(fra1gt_0,145,21). coords(fra1gt_0,191,145,21).  visible(fra1gt_0,21). happensAt(inactive(fra1gt_0),21). orientation(fra1gt_1,112,21). coords(fra1gt_1,277,16,21).  visible(fra1gt_1,21). happensAt(walking(fra1gt_1),21).
orientation(fra1gt_0,145,22). coords(fra1gt_0,193,146,22).  visible(fra1gt_0,22). happensAt(walking(fra1gt_0),22). orientation(fra1gt_1,108,22). coords(fra1gt_1,277,16,22).  visible(fra1gt_1,22). happensAt(walking(fra1gt_1),22).
orientation(fra1gt_0,145,23). coords(fra1gt_0,193,148,23).  visible(fra1gt_0,23). happensAt(walking(fra1gt_0),23). orientation(fra1gt_1,106,23). coords(fra1gt_1,277,14,23).  visible(fra1gt_1,23). happensAt(walking(fra1gt_1),23).
orientation(fra1gt_0,145,24). coords(fra1gt_0,195,149,24).  visible(fra1gt_0,24). happensAt(walking(fra1gt_0),24). orientation(fra1gt_1,106,24). coords(fra1gt_1,278,12,24).  visible(fra1gt_1,24). happensAt(walking(fra1gt_1),24).
#show holdsAt/2.