

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
orientation(wk2gt_1,143,25). coords(wk2gt_1,106,61,25).  visible(wk2gt_1,25). orientation(wk2gt_2,141,25). coords(wk2gt_2,97,67,25).  visible(wk2gt_2,25).
orientation(wk2gt_1,143,26). coords(wk2gt_1,106,60,26).  visible(wk2gt_1,26). orientation(wk2gt_2,141,26). coords(wk2gt_2,96,67,26).  visible(wk2gt_2,26).
orientation(wk2gt_1,143,27). coords(wk2gt_1,107,60,27).  visible(wk2gt_1,27). orientation(wk2gt_2,141,27). coords(wk2gt_2,97,67,27).  visible(wk2gt_2,27).
orientation(wk2gt_1,143,28). coords(wk2gt_1,108,60,28).  visible(wk2gt_1,28). orientation(wk2gt_2,141,28). coords(wk2gt_2,97,67,28).  visible(wk2gt_2,28).
orientation(wk2gt_1,143,29). coords(wk2gt_1,108,60,29).  visible(wk2gt_1,29). orientation(wk2gt_2,141,29). coords(wk2gt_2,97,66,29).  visible(wk2gt_2,29).
orientation(wk2gt_1,143,30). coords(wk2gt_1,108,60,30).  visible(wk2gt_1,30). orientation(wk2gt_2,141,30). coords(wk2gt_2,97,66,30).  visible(wk2gt_2,30).
orientation(wk2gt_1,143,31). coords(wk2gt_1,108,60,31).  visible(wk2gt_1,31). orientation(wk2gt_2,141,31). coords(wk2gt_2,97,66,31).  visible(wk2gt_2,31).
orientation(wk2gt_1,143,32). coords(wk2gt_1,108,60,32).  visible(wk2gt_1,32). orientation(wk2gt_2,141,32). coords(wk2gt_2,97,66,32).  visible(wk2gt_2,32).
orientation(wk2gt_1,143,33). coords(wk2gt_1,109,60,33).  visible(wk2gt_1,33). orientation(wk2gt_2,141,33). coords(wk2gt_2,97,66,33).  visible(wk2gt_2,33).
orientation(wk2gt_1,143,34). coords(wk2gt_1,109,59,34).  visible(wk2gt_1,34). orientation(wk2gt_2,141,34). coords(wk2gt_2,97,66,34).  visible(wk2gt_2,34).
orientation(wk2gt_1,143,35). coords(wk2gt_1,109,59,35).  visible(wk2gt_1,35). orientation(wk2gt_2,141,35). coords(wk2gt_2,97,66,35).  visible(wk2gt_2,35).
orientation(wk2gt_1,143,36). coords(wk2gt_1,109,58,36).  visible(wk2gt_1,36). orientation(wk2gt_2,141,36). coords(wk2gt_2,97,66,36).  visible(wk2gt_2,36).
orientation(wk2gt_1,143,37). coords(wk2gt_1,109,58,37).  visible(wk2gt_1,37). orientation(wk2gt_2,141,37). coords(wk2gt_2,97,66,37).  visible(wk2gt_2,37).
happensAt(disappear(wk2gt_1),36). happensAt(disappear(wk2gt_2),36).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=12..36.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=12..36.
:- holdsAt(moving(wk2gt_1,wk2gt_2),12). :- holdsAt(moving(wk2gt_2,wk2gt_1),12).
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
:- not holdsAt(moving(wk2gt_1,wk2gt_2),25). :- not holdsAt(moving(wk2gt_2,wk2gt_1),25).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),26). :- not holdsAt(moving(wk2gt_2,wk2gt_1),26).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),27). :- not holdsAt(moving(wk2gt_2,wk2gt_1),27).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),28). :- not holdsAt(moving(wk2gt_2,wk2gt_1),28).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),29). :- not holdsAt(moving(wk2gt_2,wk2gt_1),29).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),30). :- not holdsAt(moving(wk2gt_2,wk2gt_1),30).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),31). :- not holdsAt(moving(wk2gt_2,wk2gt_1),31).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),32). :- not holdsAt(moving(wk2gt_2,wk2gt_1),32).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),33). :- not holdsAt(moving(wk2gt_2,wk2gt_1),33).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),34). :- not holdsAt(moving(wk2gt_2,wk2gt_1),34).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),35). :- not holdsAt(moving(wk2gt_2,wk2gt_1),35).
:- not holdsAt(moving(wk2gt_1,wk2gt_2),36). :- not holdsAt(moving(wk2gt_2,wk2gt_1),36).
:- holdsAt(moving(wk2gt_1,wk2gt_2),37). :- holdsAt(moving(wk2gt_2,wk2gt_1),37).
:- simpleEvent(T,P,inactive), T>12 , T<36.
:- simpleEvent(T,P,running), T>12 , T<36.

#show simpleEvent/3.