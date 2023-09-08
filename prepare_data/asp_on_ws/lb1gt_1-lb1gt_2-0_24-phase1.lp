
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

	orientation(lb1gt_1,147,0). coords(lb1gt_1,72,76,0).  happensAt(appear(lb1gt_1),0). happensAt(walking(lb1gt_1),0). orientation(lb1gt_2,142,0). coords(lb1gt_2,78,63,0).  happensAt(appear(lb1gt_2),0). happensAt(walking(lb1gt_2),0).
orientation(lb1gt_1,147,1). coords(lb1gt_1,71,76,1).  visible(lb1gt_1,1). happensAt(walking(lb1gt_1),1). orientation(lb1gt_2,142,1). coords(lb1gt_2,78,63,1).  visible(lb1gt_2,1). happensAt(walking(lb1gt_2),1).
orientation(lb1gt_1,147,2). coords(lb1gt_1,71,77,2).  visible(lb1gt_1,2). happensAt(walking(lb1gt_1),2). orientation(lb1gt_2,142,2). coords(lb1gt_2,78,63,2).  visible(lb1gt_2,2). happensAt(walking(lb1gt_2),2).
orientation(lb1gt_1,147,3). coords(lb1gt_1,71,77,3).  visible(lb1gt_1,3). happensAt(walking(lb1gt_1),3). orientation(lb1gt_2,142,3). coords(lb1gt_2,78,63,3).  visible(lb1gt_2,3). happensAt(walking(lb1gt_2),3).
orientation(lb1gt_1,147,4). coords(lb1gt_1,71,78,4).  visible(lb1gt_1,4). happensAt(walking(lb1gt_1),4). orientation(lb1gt_2,142,4). coords(lb1gt_2,78,63,4).  visible(lb1gt_2,4). happensAt(walking(lb1gt_2),4).
orientation(lb1gt_1,147,5). coords(lb1gt_1,71,78,5).  visible(lb1gt_1,5). happensAt(walking(lb1gt_1),5). orientation(lb1gt_2,142,5). coords(lb1gt_2,78,64,5).  visible(lb1gt_2,5). happensAt(walking(lb1gt_2),5).
orientation(lb1gt_1,147,6). coords(lb1gt_1,71,78,6).  visible(lb1gt_1,6). happensAt(walking(lb1gt_1),6). orientation(lb1gt_2,142,6). coords(lb1gt_2,79,64,6).  visible(lb1gt_2,6). happensAt(walking(lb1gt_2),6).
orientation(lb1gt_1,147,7). coords(lb1gt_1,71,78,7).  visible(lb1gt_1,7). happensAt(walking(lb1gt_1),7). orientation(lb1gt_2,142,7). coords(lb1gt_2,79,64,7).  visible(lb1gt_2,7). happensAt(walking(lb1gt_2),7).
orientation(lb1gt_1,147,8). coords(lb1gt_1,71,78,8).  visible(lb1gt_1,8). happensAt(walking(lb1gt_1),8). orientation(lb1gt_2,142,8). coords(lb1gt_2,79,64,8).  visible(lb1gt_2,8). happensAt(walking(lb1gt_2),8).
orientation(lb1gt_1,147,9). coords(lb1gt_1,71,78,9).  visible(lb1gt_1,9). happensAt(walking(lb1gt_1),9). orientation(lb1gt_2,142,9). coords(lb1gt_2,79,65,9).  visible(lb1gt_2,9). happensAt(walking(lb1gt_2),9).
orientation(lb1gt_1,147,10). coords(lb1gt_1,71,78,10).  visible(lb1gt_1,10). happensAt(walking(lb1gt_1),10). orientation(lb1gt_2,142,10). coords(lb1gt_2,79,65,10).  visible(lb1gt_2,10). happensAt(walking(lb1gt_2),10).
orientation(lb1gt_1,147,11). coords(lb1gt_1,71,79,11).  visible(lb1gt_1,11). happensAt(walking(lb1gt_1),11). orientation(lb1gt_2,142,11). coords(lb1gt_2,79,65,11).  visible(lb1gt_2,11). happensAt(walking(lb1gt_2),11).
orientation(lb1gt_1,147,12). coords(lb1gt_1,71,79,12).  visible(lb1gt_1,12). happensAt(walking(lb1gt_1),12). orientation(lb1gt_2,142,12). coords(lb1gt_2,80,66,12).  visible(lb1gt_2,12). happensAt(walking(lb1gt_2),12).
orientation(lb1gt_1,147,13). coords(lb1gt_1,71,79,13).  visible(lb1gt_1,13). happensAt(walking(lb1gt_1),13). orientation(lb1gt_2,142,13). coords(lb1gt_2,80,66,13).  visible(lb1gt_2,13). happensAt(walking(lb1gt_2),13).
orientation(lb1gt_1,147,14). coords(lb1gt_1,71,79,14).  visible(lb1gt_1,14). happensAt(walking(lb1gt_1),14). orientation(lb1gt_2,142,14). coords(lb1gt_2,80,66,14).  visible(lb1gt_2,14). happensAt(walking(lb1gt_2),14).
orientation(lb1gt_1,147,15). coords(lb1gt_1,71,81,15).  visible(lb1gt_1,15). happensAt(walking(lb1gt_1),15). orientation(lb1gt_2,142,15). coords(lb1gt_2,80,66,15).  visible(lb1gt_2,15). happensAt(walking(lb1gt_2),15).
orientation(lb1gt_1,147,16). coords(lb1gt_1,71,81,16).  visible(lb1gt_1,16). happensAt(walking(lb1gt_1),16). orientation(lb1gt_2,142,16). coords(lb1gt_2,80,66,16).  visible(lb1gt_2,16). happensAt(walking(lb1gt_2),16).
orientation(lb1gt_1,147,17). coords(lb1gt_1,71,83,17).  visible(lb1gt_1,17). happensAt(walking(lb1gt_1),17). orientation(lb1gt_2,142,17). coords(lb1gt_2,80,67,17).  visible(lb1gt_2,17). happensAt(walking(lb1gt_2),17).
orientation(lb1gt_1,147,18). coords(lb1gt_1,71,83,18).  visible(lb1gt_1,18). happensAt(walking(lb1gt_1),18). orientation(lb1gt_2,142,18). coords(lb1gt_2,80,67,18).  visible(lb1gt_2,18). happensAt(walking(lb1gt_2),18).
orientation(lb1gt_1,147,19). coords(lb1gt_1,71,84,19).  visible(lb1gt_1,19). happensAt(walking(lb1gt_1),19). orientation(lb1gt_2,142,19). coords(lb1gt_2,80,67,19).  visible(lb1gt_2,19). happensAt(walking(lb1gt_2),19).
orientation(lb1gt_1,147,20). coords(lb1gt_1,71,85,20).  visible(lb1gt_1,20). happensAt(walking(lb1gt_1),20). orientation(lb1gt_2,142,20). coords(lb1gt_2,80,68,20).  visible(lb1gt_2,20). happensAt(walking(lb1gt_2),20).
orientation(lb1gt_1,147,21). coords(lb1gt_1,71,85,21).  visible(lb1gt_1,21). happensAt(walking(lb1gt_1),21). orientation(lb1gt_2,142,21). coords(lb1gt_2,80,69,21).  visible(lb1gt_2,21). happensAt(walking(lb1gt_2),21).
orientation(lb1gt_1,147,22). coords(lb1gt_1,71,86,22).  visible(lb1gt_1,22). happensAt(walking(lb1gt_1),22). orientation(lb1gt_2,142,22). coords(lb1gt_2,80,70,22).  visible(lb1gt_2,22). happensAt(walking(lb1gt_2),22).
orientation(lb1gt_1,147,23). coords(lb1gt_1,71,86,23).  visible(lb1gt_1,23). happensAt(walking(lb1gt_1),23). orientation(lb1gt_2,142,23). coords(lb1gt_2,80,71,23).  visible(lb1gt_2,23). happensAt(walking(lb1gt_2),23).
orientation(lb1gt_1,147,24). coords(lb1gt_1,71,86,24).  visible(lb1gt_1,24). happensAt(walking(lb1gt_1),24). orientation(lb1gt_2,142,24). coords(lb1gt_2,80,71,24).  visible(lb1gt_2,24). happensAt(walking(lb1gt_2),24).
#show holdsAt/2.