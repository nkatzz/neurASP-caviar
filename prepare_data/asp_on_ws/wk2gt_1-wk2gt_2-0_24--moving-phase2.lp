

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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).


happensAt(active(X),T) :- simpleEvent(T,X,active).
happensAt(inactive(X),T) :- simpleEvent(T,X,inactive).
happensAt(walking(X),T) :- simpleEvent(T,X,walking).
happensAt(running(X),T) :- simpleEvent(T,X,running). 


holdsByInertia(moving(X,Y),T+1) :- holdsAt(moving(X,Y),T+1), not initiatedAt(moving(X,Y),T).
holdsByInertia(meeting(X,Y),T+1) :- holdsAt(meeting(X,Y),T+1), not initiatedAt(meeting(X,Y),T).

:- holdsByInertia(moving(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2. 
:- holdsByInertia(meeting(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2.

	person(wk2gt_1;wk2gt_2).
orientation(wk2gt_1,143,0). coords(wk2gt_1,101,70,0).  happensAt(appear(wk2gt_1),0). orientation(wk2gt_2,141,0). coords(wk2gt_2,92,78,0).  happensAt(appear(wk2gt_2),0).
orientation(wk2gt_1,143,1). coords(wk2gt_1,101,70,1).  visible(wk2gt_1,1). orientation(wk2gt_2,141,1). coords(wk2gt_2,93,78,1).  visible(wk2gt_2,1).
orientation(wk2gt_1,143,2). coords(wk2gt_1,102,70,2).  visible(wk2gt_1,2). orientation(wk2gt_2,141,2). coords(wk2gt_2,93,78,2).  visible(wk2gt_2,2).
orientation(wk2gt_1,143,3). coords(wk2gt_1,102,70,3).  visible(wk2gt_1,3). orientation(wk2gt_2,141,3). coords(wk2gt_2,93,78,3).  visible(wk2gt_2,3).
orientation(wk2gt_1,143,4). coords(wk2gt_1,102,69,4).  visible(wk2gt_1,4). orientation(wk2gt_2,141,4). coords(wk2gt_2,93,78,4).  visible(wk2gt_2,4).
orientation(wk2gt_1,143,5). coords(wk2gt_1,103,69,5).  visible(wk2gt_1,5). orientation(wk2gt_2,141,5). coords(wk2gt_2,93,77,5).  visible(wk2gt_2,5).
orientation(wk2gt_1,143,6). coords(wk2gt_1,102,68,6).  visible(wk2gt_1,6). orientation(wk2gt_2,141,6). coords(wk2gt_2,94,77,6).  visible(wk2gt_2,6).
orientation(wk2gt_1,143,7). coords(wk2gt_1,102,67,7).  visible(wk2gt_1,7). orientation(wk2gt_2,141,7). coords(wk2gt_2,94,77,7).  visible(wk2gt_2,7).
orientation(wk2gt_1,143,8). coords(wk2gt_1,103,67,8).  visible(wk2gt_1,8). orientation(wk2gt_2,141,8). coords(wk2gt_2,94,77,8).  visible(wk2gt_2,8).
orientation(wk2gt_1,143,9). coords(wk2gt_1,103,67,9).  visible(wk2gt_1,9). orientation(wk2gt_2,141,9). coords(wk2gt_2,94,77,9).  visible(wk2gt_2,9).
orientation(wk2gt_1,143,10). coords(wk2gt_1,103,66,10).  visible(wk2gt_1,10). orientation(wk2gt_2,141,10). coords(wk2gt_2,95,77,10).  visible(wk2gt_2,10).
orientation(wk2gt_1,143,11). coords(wk2gt_1,103,66,11).  visible(wk2gt_1,11). orientation(wk2gt_2,141,11). coords(wk2gt_2,96,77,11).  visible(wk2gt_2,11).
orientation(wk2gt_1,143,12). coords(wk2gt_1,103,65,12).  visible(wk2gt_1,12). orientation(wk2gt_2,141,12). coords(wk2gt_2,95,76,12).  visible(wk2gt_2,12).
orientation(wk2gt_1,143,13). coords(wk2gt_1,103,65,13).  visible(wk2gt_1,13). orientation(wk2gt_2,141,13). coords(wk2gt_2,95,75,13).  visible(wk2gt_2,13).
orientation(wk2gt_1,143,14). coords(wk2gt_1,104,65,14).  visible(wk2gt_1,14). orientation(wk2gt_2,141,14). coords(wk2gt_2,95,74,14).  visible(wk2gt_2,14).
orientation(wk2gt_1,143,15). coords(wk2gt_1,104,65,15).  visible(wk2gt_1,15). orientation(wk2gt_2,141,15). coords(wk2gt_2,95,73,15).  visible(wk2gt_2,15).
orientation(wk2gt_1,143,16). coords(wk2gt_1,104,65,16).  visible(wk2gt_1,16). orientation(wk2gt_2,141,16). coords(wk2gt_2,95,73,16).  visible(wk2gt_2,16).
orientation(wk2gt_1,143,17). coords(wk2gt_1,104,65,17).  visible(wk2gt_1,17). orientation(wk2gt_2,141,17). coords(wk2gt_2,95,73,17).  visible(wk2gt_2,17).
orientation(wk2gt_1,143,18). coords(wk2gt_1,105,64,18).  visible(wk2gt_1,18). orientation(wk2gt_2,141,18). coords(wk2gt_2,95,72,18).  visible(wk2gt_2,18).
orientation(wk2gt_1,143,19). coords(wk2gt_1,105,64,19).  visible(wk2gt_1,19). orientation(wk2gt_2,141,19). coords(wk2gt_2,96,71,19).  visible(wk2gt_2,19).
orientation(wk2gt_1,143,20). coords(wk2gt_1,105,64,20).  visible(wk2gt_1,20). orientation(wk2gt_2,141,20). coords(wk2gt_2,96,70,20).  visible(wk2gt_2,20).
orientation(wk2gt_1,143,21). coords(wk2gt_1,105,63,21).  visible(wk2gt_1,21). orientation(wk2gt_2,141,21). coords(wk2gt_2,96,70,21).  visible(wk2gt_2,21).
orientation(wk2gt_1,143,22). coords(wk2gt_1,105,62,22).  visible(wk2gt_1,22). orientation(wk2gt_2,141,22). coords(wk2gt_2,96,69,22).  visible(wk2gt_2,22).
orientation(wk2gt_1,143,23). coords(wk2gt_1,105,62,23).  visible(wk2gt_1,23). orientation(wk2gt_2,141,23). coords(wk2gt_2,97,68,23).  visible(wk2gt_2,23).
orientation(wk2gt_1,143,24). coords(wk2gt_1,105,62,24).  visible(wk2gt_1,24). orientation(wk2gt_2,141,24). coords(wk2gt_2,97,68,24).  visible(wk2gt_2,24).
orientation(wk2gt_1,143,25). coords(wk2gt_1,105,62,25).  visible(wk2gt_1,25). orientation(wk2gt_2,141,25). coords(wk2gt_2,97,68,25).  visible(wk2gt_2,25).
happensAt(disappear(wk2gt_1),24). happensAt(disappear(wk2gt_2),24).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=0..24.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=0..24.
:- holdsAt(moving(wk2gt_1,wk2gt_2),0). :- holdsAt(moving(wk2gt_2,wk2gt_1),0).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),1). :- not holdsAt(moving(wk2gt_2,wk2gt_1),1).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),2). :- not holdsAt(moving(wk2gt_2,wk2gt_1),2).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),3). :- not holdsAt(moving(wk2gt_2,wk2gt_1),3).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),4). :- not holdsAt(moving(wk2gt_2,wk2gt_1),4).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),5). :- not holdsAt(moving(wk2gt_2,wk2gt_1),5).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),6). :- not holdsAt(moving(wk2gt_2,wk2gt_1),6).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),7). :- not holdsAt(moving(wk2gt_2,wk2gt_1),7).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),8). :- not holdsAt(moving(wk2gt_2,wk2gt_1),8).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),9). :- not holdsAt(moving(wk2gt_2,wk2gt_1),9).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),10). :- not holdsAt(moving(wk2gt_2,wk2gt_1),10).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),11). :- not holdsAt(moving(wk2gt_2,wk2gt_1),11).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),12). :- not holdsAt(moving(wk2gt_2,wk2gt_1),12).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),13). :- not holdsAt(moving(wk2gt_2,wk2gt_1),13).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),14). :- not holdsAt(moving(wk2gt_2,wk2gt_1),14).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),15). :- not holdsAt(moving(wk2gt_2,wk2gt_1),15).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),16). :- not holdsAt(moving(wk2gt_2,wk2gt_1),16).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),17). :- not holdsAt(moving(wk2gt_2,wk2gt_1),17).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),18). :- not holdsAt(moving(wk2gt_2,wk2gt_1),18).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),19). :- not holdsAt(moving(wk2gt_2,wk2gt_1),19).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),20). :- not holdsAt(moving(wk2gt_2,wk2gt_1),20).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),21). :- not holdsAt(moving(wk2gt_2,wk2gt_1),21).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),22). :- not holdsAt(moving(wk2gt_2,wk2gt_1),22).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),23). :- not holdsAt(moving(wk2gt_2,wk2gt_1),23).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),24). :- not holdsAt(moving(wk2gt_2,wk2gt_1),24).
:- holdsAt(moving(wk2gt_1,wk2gt_2),25). :- holdsAt(moving(wk2gt_2,wk2gt_1),25).
:- simpleEvent(T,P,inactive), T>0 , T<24.
:- simpleEvent(T,P,running), T>0 , T<24.

#show simpleEvent/3.