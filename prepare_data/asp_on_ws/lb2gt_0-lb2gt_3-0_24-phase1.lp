
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

	orientation(lb2gt_0,141,0). coords(lb2gt_0,189,125,0).  happensAt(appear(lb2gt_0),0). happensAt(inactive(lb2gt_0),0). orientation(lb2gt_3,146,0). coords(lb2gt_3,30,29,0).  happensAt(appear(lb2gt_3),0). happensAt(walking(lb2gt_3),0).
orientation(lb2gt_0,141,1). coords(lb2gt_0,189,125,1).  visible(lb2gt_0,1). happensAt(inactive(lb2gt_0),1). orientation(lb2gt_3,146,1). coords(lb2gt_3,29,29,1).  visible(lb2gt_3,1). happensAt(walking(lb2gt_3),1).
orientation(lb2gt_0,141,2). coords(lb2gt_0,189,125,2).  visible(lb2gt_0,2). happensAt(inactive(lb2gt_0),2). orientation(lb2gt_3,146,2). coords(lb2gt_3,29,29,2).  visible(lb2gt_3,2). happensAt(walking(lb2gt_3),2).
orientation(lb2gt_0,141,3). coords(lb2gt_0,189,125,3).  visible(lb2gt_0,3). happensAt(inactive(lb2gt_0),3). orientation(lb2gt_3,146,3). coords(lb2gt_3,29,29,3).  visible(lb2gt_3,3). happensAt(walking(lb2gt_3),3).
orientation(lb2gt_0,141,4). coords(lb2gt_0,189,125,4).  visible(lb2gt_0,4). happensAt(inactive(lb2gt_0),4). orientation(lb2gt_3,146,4). coords(lb2gt_3,29,29,4).  visible(lb2gt_3,4). happensAt(walking(lb2gt_3),4).
orientation(lb2gt_0,141,5). coords(lb2gt_0,189,125,5).  visible(lb2gt_0,5). happensAt(inactive(lb2gt_0),5). orientation(lb2gt_3,146,5). coords(lb2gt_3,29,29,5).  visible(lb2gt_3,5). happensAt(walking(lb2gt_3),5).
orientation(lb2gt_0,141,6). coords(lb2gt_0,189,125,6).  visible(lb2gt_0,6). happensAt(inactive(lb2gt_0),6). orientation(lb2gt_3,146,6). coords(lb2gt_3,29,29,6).  visible(lb2gt_3,6). happensAt(walking(lb2gt_3),6).
orientation(lb2gt_0,141,7). coords(lb2gt_0,190,125,7).  visible(lb2gt_0,7). happensAt(inactive(lb2gt_0),7). orientation(lb2gt_3,146,7). coords(lb2gt_3,29,29,7).  visible(lb2gt_3,7). happensAt(walking(lb2gt_3),7).
orientation(lb2gt_0,141,8). coords(lb2gt_0,190,125,8).  visible(lb2gt_0,8). happensAt(inactive(lb2gt_0),8). orientation(lb2gt_3,146,8). coords(lb2gt_3,29,30,8).  visible(lb2gt_3,8). happensAt(walking(lb2gt_3),8).
orientation(lb2gt_0,141,9). coords(lb2gt_0,190,125,9).  visible(lb2gt_0,9). happensAt(inactive(lb2gt_0),9). orientation(lb2gt_3,146,9). coords(lb2gt_3,29,30,9).  visible(lb2gt_3,9). happensAt(walking(lb2gt_3),9).
orientation(lb2gt_0,141,10). coords(lb2gt_0,190,125,10).  visible(lb2gt_0,10). happensAt(inactive(lb2gt_0),10). orientation(lb2gt_3,146,10). coords(lb2gt_3,29,31,10).  visible(lb2gt_3,10). happensAt(walking(lb2gt_3),10).
orientation(lb2gt_0,141,11). coords(lb2gt_0,190,125,11).  visible(lb2gt_0,11). happensAt(inactive(lb2gt_0),11). orientation(lb2gt_3,146,11). coords(lb2gt_3,29,31,11).  visible(lb2gt_3,11). happensAt(walking(lb2gt_3),11).
orientation(lb2gt_0,141,12). coords(lb2gt_0,190,125,12).  visible(lb2gt_0,12). happensAt(inactive(lb2gt_0),12). orientation(lb2gt_3,146,12). coords(lb2gt_3,29,31,12).  visible(lb2gt_3,12). happensAt(walking(lb2gt_3),12).
orientation(lb2gt_0,141,13). coords(lb2gt_0,190,125,13).  visible(lb2gt_0,13). happensAt(inactive(lb2gt_0),13). orientation(lb2gt_3,146,13). coords(lb2gt_3,29,30,13).  visible(lb2gt_3,13). happensAt(walking(lb2gt_3),13).
orientation(lb2gt_0,141,14). coords(lb2gt_0,190,125,14).  visible(lb2gt_0,14). happensAt(inactive(lb2gt_0),14). orientation(lb2gt_3,146,14). coords(lb2gt_3,29,31,14).  visible(lb2gt_3,14). happensAt(walking(lb2gt_3),14).
orientation(lb2gt_0,141,15). coords(lb2gt_0,190,125,15).  visible(lb2gt_0,15). happensAt(inactive(lb2gt_0),15). orientation(lb2gt_3,146,15). coords(lb2gt_3,29,31,15).  visible(lb2gt_3,15). happensAt(walking(lb2gt_3),15).
orientation(lb2gt_0,141,16). coords(lb2gt_0,190,125,16).  visible(lb2gt_0,16). happensAt(inactive(lb2gt_0),16). orientation(lb2gt_3,146,16). coords(lb2gt_3,29,32,16).  visible(lb2gt_3,16). happensAt(walking(lb2gt_3),16).
orientation(lb2gt_0,141,17). coords(lb2gt_0,190,125,17).  visible(lb2gt_0,17). happensAt(inactive(lb2gt_0),17). orientation(lb2gt_3,146,17). coords(lb2gt_3,27,32,17).  visible(lb2gt_3,17). happensAt(walking(lb2gt_3),17).
orientation(lb2gt_0,141,18). coords(lb2gt_0,190,125,18).  visible(lb2gt_0,18). happensAt(inactive(lb2gt_0),18). orientation(lb2gt_3,146,18). coords(lb2gt_3,27,32,18).  visible(lb2gt_3,18). happensAt(walking(lb2gt_3),18).
orientation(lb2gt_0,141,19). coords(lb2gt_0,190,125,19).  visible(lb2gt_0,19). happensAt(inactive(lb2gt_0),19). orientation(lb2gt_3,146,19). coords(lb2gt_3,26,33,19).  visible(lb2gt_3,19). happensAt(walking(lb2gt_3),19).
orientation(lb2gt_0,141,20). coords(lb2gt_0,190,125,20).  visible(lb2gt_0,20). happensAt(inactive(lb2gt_0),20). orientation(lb2gt_3,146,20). coords(lb2gt_3,25,33,20).  visible(lb2gt_3,20). happensAt(walking(lb2gt_3),20).
orientation(lb2gt_0,141,21). coords(lb2gt_0,190,125,21).  visible(lb2gt_0,21). happensAt(inactive(lb2gt_0),21). orientation(lb2gt_3,146,21). coords(lb2gt_3,25,33,21).  visible(lb2gt_3,21). happensAt(walking(lb2gt_3),21).
orientation(lb2gt_0,141,22). coords(lb2gt_0,190,125,22).  visible(lb2gt_0,22). happensAt(inactive(lb2gt_0),22). orientation(lb2gt_3,146,22). coords(lb2gt_3,24,33,22).  visible(lb2gt_3,22). happensAt(walking(lb2gt_3),22).
orientation(lb2gt_0,141,23). coords(lb2gt_0,190,125,23).  visible(lb2gt_0,23). happensAt(inactive(lb2gt_0),23). orientation(lb2gt_3,146,23). coords(lb2gt_3,24,34,23).  visible(lb2gt_3,23). happensAt(walking(lb2gt_3),23).
orientation(lb2gt_0,141,24). coords(lb2gt_0,190,125,24).  visible(lb2gt_0,24). happensAt(inactive(lb2gt_0),24). orientation(lb2gt_3,146,24). coords(lb2gt_3,23,34,24).  visible(lb2gt_3,24). happensAt(walking(lb2gt_3),24).
#show holdsAt/2.