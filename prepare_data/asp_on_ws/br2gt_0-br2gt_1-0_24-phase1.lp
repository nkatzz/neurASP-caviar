
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

	orientation(br2gt_0,140,0). coords(br2gt_0,77,75,0).  happensAt(appear(br2gt_0),0). happensAt(walking(br2gt_0),0). orientation(br2gt_1,131,0). coords(br2gt_1,305,283,0).  happensAt(appear(br2gt_1),0). happensAt(walking(br2gt_1),0).
orientation(br2gt_0,140,1). coords(br2gt_0,77,75,1).  visible(br2gt_0,1). happensAt(walking(br2gt_0),1). orientation(br2gt_1,131,1). coords(br2gt_1,305,283,1).  visible(br2gt_1,1). happensAt(walking(br2gt_1),1).
orientation(br2gt_0,133,2). coords(br2gt_0,77,74,2).  visible(br2gt_0,2). happensAt(walking(br2gt_0),2). orientation(br2gt_1,131,2). coords(br2gt_1,305,283,2).  visible(br2gt_1,2). happensAt(walking(br2gt_1),2).
orientation(br2gt_0,133,3). coords(br2gt_0,77,73,3).  visible(br2gt_0,3). happensAt(walking(br2gt_0),3). orientation(br2gt_1,131,3). coords(br2gt_1,305,283,3).  visible(br2gt_1,3). happensAt(walking(br2gt_1),3).
orientation(br2gt_0,133,4). coords(br2gt_0,77,73,4).  visible(br2gt_0,4). happensAt(walking(br2gt_0),4). orientation(br2gt_1,131,4). coords(br2gt_1,305,283,4).  visible(br2gt_1,4). happensAt(walking(br2gt_1),4).
orientation(br2gt_0,133,5). coords(br2gt_0,77,72,5).  visible(br2gt_0,5). happensAt(walking(br2gt_0),5). orientation(br2gt_1,131,5). coords(br2gt_1,305,283,5).  visible(br2gt_1,5). happensAt(walking(br2gt_1),5).
orientation(br2gt_0,133,6). coords(br2gt_0,77,71,6).  visible(br2gt_0,6). happensAt(walking(br2gt_0),6). orientation(br2gt_1,131,6). coords(br2gt_1,305,283,6).  visible(br2gt_1,6). happensAt(walking(br2gt_1),6).
orientation(br2gt_0,133,7). coords(br2gt_0,77,70,7).  visible(br2gt_0,7). happensAt(walking(br2gt_0),7). orientation(br2gt_1,131,7). coords(br2gt_1,305,283,7).  visible(br2gt_1,7). happensAt(walking(br2gt_1),7).
orientation(br2gt_0,133,8). coords(br2gt_0,77,70,8).  visible(br2gt_0,8). happensAt(walking(br2gt_0),8). orientation(br2gt_1,131,8). coords(br2gt_1,305,283,8).  visible(br2gt_1,8). happensAt(walking(br2gt_1),8).
orientation(br2gt_0,133,9). coords(br2gt_0,77,67,9).  visible(br2gt_0,9). happensAt(walking(br2gt_0),9). orientation(br2gt_1,131,9). coords(br2gt_1,305,283,9).  visible(br2gt_1,9). happensAt(walking(br2gt_1),9).
orientation(br2gt_0,133,10). coords(br2gt_0,77,67,10).  visible(br2gt_0,10). happensAt(walking(br2gt_0),10). orientation(br2gt_1,131,10). coords(br2gt_1,305,283,10).  visible(br2gt_1,10). happensAt(walking(br2gt_1),10).
orientation(br2gt_0,133,11). coords(br2gt_0,77,66,11).  visible(br2gt_0,11). happensAt(walking(br2gt_0),11). orientation(br2gt_1,131,11). coords(br2gt_1,305,283,11).  visible(br2gt_1,11). happensAt(walking(br2gt_1),11).
orientation(br2gt_0,133,12). coords(br2gt_0,77,66,12).  visible(br2gt_0,12). happensAt(walking(br2gt_0),12). orientation(br2gt_1,131,12). coords(br2gt_1,305,283,12).  visible(br2gt_1,12). happensAt(walking(br2gt_1),12).
orientation(br2gt_0,133,13). coords(br2gt_0,77,64,13).  visible(br2gt_0,13). happensAt(walking(br2gt_0),13). orientation(br2gt_1,131,13). coords(br2gt_1,305,283,13).  visible(br2gt_1,13). happensAt(walking(br2gt_1),13).
orientation(br2gt_0,133,14). coords(br2gt_0,77,64,14).  visible(br2gt_0,14). happensAt(walking(br2gt_0),14). orientation(br2gt_1,131,14). coords(br2gt_1,303,283,14).  visible(br2gt_1,14). happensAt(walking(br2gt_1),14).
orientation(br2gt_0,133,15). coords(br2gt_0,77,64,15).  visible(br2gt_0,15). happensAt(walking(br2gt_0),15). orientation(br2gt_1,131,15). coords(br2gt_1,302,283,15).  visible(br2gt_1,15). happensAt(walking(br2gt_1),15).
orientation(br2gt_0,133,16). coords(br2gt_0,77,63,16).  visible(br2gt_0,16). happensAt(walking(br2gt_0),16). orientation(br2gt_1,131,16). coords(br2gt_1,302,283,16).  visible(br2gt_1,16). happensAt(walking(br2gt_1),16).
orientation(br2gt_0,133,17). coords(br2gt_0,78,63,17).  visible(br2gt_0,17). happensAt(walking(br2gt_0),17). orientation(br2gt_1,131,17). coords(br2gt_1,302,283,17).  visible(br2gt_1,17). happensAt(walking(br2gt_1),17).
orientation(br2gt_0,133,18). coords(br2gt_0,78,63,18).  visible(br2gt_0,18). happensAt(walking(br2gt_0),18). orientation(br2gt_1,131,18). coords(br2gt_1,286,282,18).  visible(br2gt_1,18). happensAt(walking(br2gt_1),18).
orientation(br2gt_0,133,19). coords(br2gt_0,78,62,19).  visible(br2gt_0,19). happensAt(walking(br2gt_0),19). orientation(br2gt_1,131,19). coords(br2gt_1,284,280,19).  visible(br2gt_1,19). happensAt(walking(br2gt_1),19).
orientation(br2gt_0,133,20). coords(br2gt_0,78,61,20).  visible(br2gt_0,20). happensAt(walking(br2gt_0),20). orientation(br2gt_1,131,20). coords(br2gt_1,283,279,20).  visible(br2gt_1,20). happensAt(walking(br2gt_1),20).
orientation(br2gt_0,133,21). coords(br2gt_0,79,61,21).  visible(br2gt_0,21). happensAt(walking(br2gt_0),21). orientation(br2gt_1,172,21). coords(br2gt_1,283,277,21).  visible(br2gt_1,21). happensAt(walking(br2gt_1),21).
orientation(br2gt_0,133,22). coords(br2gt_0,79,61,22).  visible(br2gt_0,22). happensAt(walking(br2gt_0),22). orientation(br2gt_1,175,22). coords(br2gt_1,281,276,22).  visible(br2gt_1,22). happensAt(walking(br2gt_1),22).
orientation(br2gt_0,133,23). coords(br2gt_0,79,61,23).  visible(br2gt_0,23). happensAt(walking(br2gt_0),23). orientation(br2gt_1,175,23). coords(br2gt_1,280,275,23).  visible(br2gt_1,23). happensAt(walking(br2gt_1),23).
orientation(br2gt_0,133,24). coords(br2gt_0,79,61,24).  visible(br2gt_0,24). happensAt(walking(br2gt_0),24). orientation(br2gt_1,175,24). coords(br2gt_1,278,275,24).  visible(br2gt_1,24). happensAt(walking(br2gt_1),24).
#show holdsAt/2.