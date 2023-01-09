

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

	person(lb2gt_2;lb2gt_3).
orientation(lb2gt_2,147,0). coords(lb2gt_2,26,48,0).  happensAt(appear(lb2gt_2),0). orientation(lb2gt_3,146,0). coords(lb2gt_3,30,29,0).  happensAt(appear(lb2gt_3),0).
orientation(lb2gt_2,147,1). coords(lb2gt_2,25,48,1).  visible(lb2gt_2,1). orientation(lb2gt_3,146,1). coords(lb2gt_3,29,29,1).  visible(lb2gt_3,1).
orientation(lb2gt_2,147,2). coords(lb2gt_2,25,48,2).  visible(lb2gt_2,2). orientation(lb2gt_3,146,2). coords(lb2gt_3,29,29,2).  visible(lb2gt_3,2).
orientation(lb2gt_2,147,3). coords(lb2gt_2,25,48,3).  visible(lb2gt_2,3). orientation(lb2gt_3,146,3). coords(lb2gt_3,29,29,3).  visible(lb2gt_3,3).
orientation(lb2gt_2,147,4). coords(lb2gt_2,25,48,4).  visible(lb2gt_2,4). orientation(lb2gt_3,146,4). coords(lb2gt_3,29,29,4).  visible(lb2gt_3,4).
orientation(lb2gt_2,147,5). coords(lb2gt_2,23,48,5).  visible(lb2gt_2,5). orientation(lb2gt_3,146,5). coords(lb2gt_3,29,29,5).  visible(lb2gt_3,5).
orientation(lb2gt_2,147,6). coords(lb2gt_2,22,49,6).  visible(lb2gt_2,6). orientation(lb2gt_3,146,6). coords(lb2gt_3,29,29,6).  visible(lb2gt_3,6).
orientation(lb2gt_2,147,7). coords(lb2gt_2,22,49,7).  visible(lb2gt_2,7). orientation(lb2gt_3,146,7). coords(lb2gt_3,29,29,7).  visible(lb2gt_3,7).
orientation(lb2gt_2,147,8). coords(lb2gt_2,22,49,8).  visible(lb2gt_2,8). orientation(lb2gt_3,146,8). coords(lb2gt_3,29,30,8).  visible(lb2gt_3,8).
orientation(lb2gt_2,147,9). coords(lb2gt_2,22,49,9).  visible(lb2gt_2,9). orientation(lb2gt_3,146,9). coords(lb2gt_3,29,30,9).  visible(lb2gt_3,9).
orientation(lb2gt_2,147,10). coords(lb2gt_2,22,49,10).  visible(lb2gt_2,10). orientation(lb2gt_3,146,10). coords(lb2gt_3,29,31,10).  visible(lb2gt_3,10).
orientation(lb2gt_2,147,11). coords(lb2gt_2,22,49,11).  visible(lb2gt_2,11). orientation(lb2gt_3,146,11). coords(lb2gt_3,29,31,11).  visible(lb2gt_3,11).
orientation(lb2gt_2,147,12). coords(lb2gt_2,22,49,12).  visible(lb2gt_2,12). orientation(lb2gt_3,146,12). coords(lb2gt_3,29,31,12).  visible(lb2gt_3,12).
orientation(lb2gt_2,147,13). coords(lb2gt_2,22,49,13).  visible(lb2gt_2,13). orientation(lb2gt_3,146,13). coords(lb2gt_3,29,30,13).  visible(lb2gt_3,13).
orientation(lb2gt_2,147,14). coords(lb2gt_2,22,49,14).  visible(lb2gt_2,14). orientation(lb2gt_3,146,14). coords(lb2gt_3,29,31,14).  visible(lb2gt_3,14).
orientation(lb2gt_2,147,15). coords(lb2gt_2,21,49,15).  visible(lb2gt_2,15). orientation(lb2gt_3,146,15). coords(lb2gt_3,29,31,15).  visible(lb2gt_3,15).
orientation(lb2gt_2,147,16). coords(lb2gt_2,21,49,16).  visible(lb2gt_2,16). orientation(lb2gt_3,146,16). coords(lb2gt_3,29,32,16).  visible(lb2gt_3,16).
orientation(lb2gt_2,147,17). coords(lb2gt_2,21,49,17).  visible(lb2gt_2,17). orientation(lb2gt_3,146,17). coords(lb2gt_3,27,32,17).  visible(lb2gt_3,17).
orientation(lb2gt_2,147,18). coords(lb2gt_2,21,49,18).  visible(lb2gt_2,18). orientation(lb2gt_3,146,18). coords(lb2gt_3,27,32,18).  visible(lb2gt_3,18).
orientation(lb2gt_2,147,19). coords(lb2gt_2,21,50,19).  visible(lb2gt_2,19). orientation(lb2gt_3,146,19). coords(lb2gt_3,26,33,19).  visible(lb2gt_3,19).
orientation(lb2gt_2,147,20). coords(lb2gt_2,21,50,20).  visible(lb2gt_2,20). orientation(lb2gt_3,146,20). coords(lb2gt_3,25,33,20).  visible(lb2gt_3,20).
orientation(lb2gt_2,147,21). coords(lb2gt_2,21,50,21).  visible(lb2gt_2,21). orientation(lb2gt_3,146,21). coords(lb2gt_3,25,33,21).  visible(lb2gt_3,21).
orientation(lb2gt_2,147,22). coords(lb2gt_2,21,50,22).  visible(lb2gt_2,22). orientation(lb2gt_3,146,22). coords(lb2gt_3,24,33,22).  visible(lb2gt_3,22).
orientation(lb2gt_2,147,23). coords(lb2gt_2,21,50,23).  visible(lb2gt_2,23). orientation(lb2gt_3,146,23). coords(lb2gt_3,24,34,23).  visible(lb2gt_3,23).
orientation(lb2gt_2,147,24). coords(lb2gt_2,21,50,24).  visible(lb2gt_2,24). orientation(lb2gt_3,146,24). coords(lb2gt_3,23,34,24).  visible(lb2gt_3,24).
orientation(lb2gt_2,147,25). coords(lb2gt_2,21,50,25).  visible(lb2gt_2,25). orientation(lb2gt_3,146,25). coords(lb2gt_3,23,34,25).  visible(lb2gt_3,25).
happensAt(disappear(lb2gt_2),24). happensAt(disappear(lb2gt_3),24).
1{simpleEvent(T,lb2gt_2,active); simpleEvent(T,lb2gt_2,inactive); simpleEvent(T,lb2gt_2,walking); simpleEvent(T,lb2gt_2,running)}1 :- T=0..24.
1{simpleEvent(T,lb2gt_3,active); simpleEvent(T,lb2gt_3,inactive); simpleEvent(T,lb2gt_3,walking); simpleEvent(T,lb2gt_3,running)}1 :- T=0..24.
:- holdsAt(moving(lb2gt_2,lb2gt_3),0). :- holdsAt(moving(lb2gt_3,lb2gt_2),0).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),1). :- not holdsAt(moving(lb2gt_3,lb2gt_2),1).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),2). :- not holdsAt(moving(lb2gt_3,lb2gt_2),2).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),3). :- not holdsAt(moving(lb2gt_3,lb2gt_2),3).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),4). :- not holdsAt(moving(lb2gt_3,lb2gt_2),4).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),5). :- not holdsAt(moving(lb2gt_3,lb2gt_2),5).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),6). :- not holdsAt(moving(lb2gt_3,lb2gt_2),6).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),7). :- not holdsAt(moving(lb2gt_3,lb2gt_2),7).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),8). :- not holdsAt(moving(lb2gt_3,lb2gt_2),8).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),9). :- not holdsAt(moving(lb2gt_3,lb2gt_2),9).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),10). :- not holdsAt(moving(lb2gt_3,lb2gt_2),10).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),11). :- not holdsAt(moving(lb2gt_3,lb2gt_2),11).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),12). :- not holdsAt(moving(lb2gt_3,lb2gt_2),12).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),13). :- not holdsAt(moving(lb2gt_3,lb2gt_2),13).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),14). :- not holdsAt(moving(lb2gt_3,lb2gt_2),14).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),15). :- not holdsAt(moving(lb2gt_3,lb2gt_2),15).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),16). :- not holdsAt(moving(lb2gt_3,lb2gt_2),16).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),17). :- not holdsAt(moving(lb2gt_3,lb2gt_2),17).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),18). :- not holdsAt(moving(lb2gt_3,lb2gt_2),18).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),19). :- not holdsAt(moving(lb2gt_3,lb2gt_2),19).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),20). :- not holdsAt(moving(lb2gt_3,lb2gt_2),20).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),21). :- not holdsAt(moving(lb2gt_3,lb2gt_2),21).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),22). :- not holdsAt(moving(lb2gt_3,lb2gt_2),22).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),23). :- not holdsAt(moving(lb2gt_3,lb2gt_2),23).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),24). :- not holdsAt(moving(lb2gt_3,lb2gt_2),24).
:- holdsAt(moving(lb2gt_2,lb2gt_3),25). :- holdsAt(moving(lb2gt_3,lb2gt_2),25).
:- simpleEvent(T,P,inactive), T>0 , T<24.
:- simpleEvent(T,P,running), T>0 , T<24.

#show simpleEvent/3.