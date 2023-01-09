

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

	person(lb1gt_1;lb1gt_2).
orientation(lb1gt_1,147,12). coords(lb1gt_1,71,79,12).  visible(lb1gt_1,12). orientation(lb1gt_2,142,12). coords(lb1gt_2,80,66,12).  visible(lb1gt_2,12).
orientation(lb1gt_1,147,13). coords(lb1gt_1,71,79,13).  visible(lb1gt_1,13). orientation(lb1gt_2,142,13). coords(lb1gt_2,80,66,13).  visible(lb1gt_2,13).
orientation(lb1gt_1,147,14). coords(lb1gt_1,71,79,14).  visible(lb1gt_1,14). orientation(lb1gt_2,142,14). coords(lb1gt_2,80,66,14).  visible(lb1gt_2,14).
orientation(lb1gt_1,147,15). coords(lb1gt_1,71,81,15).  visible(lb1gt_1,15). orientation(lb1gt_2,142,15). coords(lb1gt_2,80,66,15).  visible(lb1gt_2,15).
orientation(lb1gt_1,147,16). coords(lb1gt_1,71,81,16).  visible(lb1gt_1,16). orientation(lb1gt_2,142,16). coords(lb1gt_2,80,66,16).  visible(lb1gt_2,16).
orientation(lb1gt_1,147,17). coords(lb1gt_1,71,83,17).  visible(lb1gt_1,17). orientation(lb1gt_2,142,17). coords(lb1gt_2,80,67,17).  visible(lb1gt_2,17).
orientation(lb1gt_1,147,18). coords(lb1gt_1,71,83,18).  visible(lb1gt_1,18). orientation(lb1gt_2,142,18). coords(lb1gt_2,80,67,18).  visible(lb1gt_2,18).
orientation(lb1gt_1,147,19). coords(lb1gt_1,71,84,19).  visible(lb1gt_1,19). orientation(lb1gt_2,142,19). coords(lb1gt_2,80,67,19).  visible(lb1gt_2,19).
orientation(lb1gt_1,147,20). coords(lb1gt_1,71,85,20).  visible(lb1gt_1,20). orientation(lb1gt_2,142,20). coords(lb1gt_2,80,68,20).  visible(lb1gt_2,20).
orientation(lb1gt_1,147,21). coords(lb1gt_1,71,85,21).  visible(lb1gt_1,21). orientation(lb1gt_2,142,21). coords(lb1gt_2,80,69,21).  visible(lb1gt_2,21).
orientation(lb1gt_1,147,22). coords(lb1gt_1,71,86,22).  visible(lb1gt_1,22). orientation(lb1gt_2,142,22). coords(lb1gt_2,80,70,22).  visible(lb1gt_2,22).
orientation(lb1gt_1,147,23). coords(lb1gt_1,71,86,23).  visible(lb1gt_1,23). orientation(lb1gt_2,142,23). coords(lb1gt_2,80,71,23).  visible(lb1gt_2,23).
orientation(lb1gt_1,147,24). coords(lb1gt_1,71,86,24).  visible(lb1gt_1,24). orientation(lb1gt_2,142,24). coords(lb1gt_2,80,71,24).  visible(lb1gt_2,24).
orientation(lb1gt_1,147,25). coords(lb1gt_1,71,86,25).  visible(lb1gt_1,25). orientation(lb1gt_2,142,25). coords(lb1gt_2,80,72,25).  visible(lb1gt_2,25).
orientation(lb1gt_1,147,26). coords(lb1gt_1,71,87,26).  visible(lb1gt_1,26). orientation(lb1gt_2,142,26). coords(lb1gt_2,80,72,26).  visible(lb1gt_2,26).
orientation(lb1gt_1,147,27). coords(lb1gt_1,71,87,27).  visible(lb1gt_1,27). orientation(lb1gt_2,142,27). coords(lb1gt_2,80,73,27).  visible(lb1gt_2,27).
orientation(lb1gt_1,147,28). coords(lb1gt_1,71,87,28).  visible(lb1gt_1,28). orientation(lb1gt_2,142,28). coords(lb1gt_2,79,73,28).  visible(lb1gt_2,28).
orientation(lb1gt_1,147,29). coords(lb1gt_1,71,87,29).  visible(lb1gt_1,29). orientation(lb1gt_2,142,29). coords(lb1gt_2,79,74,29).  visible(lb1gt_2,29).
orientation(lb1gt_1,147,30). coords(lb1gt_1,71,88,30).  visible(lb1gt_1,30). orientation(lb1gt_2,142,30). coords(lb1gt_2,79,74,30).  visible(lb1gt_2,30).
orientation(lb1gt_1,147,31). coords(lb1gt_1,71,88,31).  visible(lb1gt_1,31). orientation(lb1gt_2,142,31). coords(lb1gt_2,79,74,31).  visible(lb1gt_2,31).
orientation(lb1gt_1,147,32). coords(lb1gt_1,72,88,32).  visible(lb1gt_1,32). orientation(lb1gt_2,142,32). coords(lb1gt_2,79,74,32).  visible(lb1gt_2,32).
orientation(lb1gt_1,147,33). coords(lb1gt_1,72,88,33).  visible(lb1gt_1,33). orientation(lb1gt_2,142,33). coords(lb1gt_2,79,74,33).  visible(lb1gt_2,33).
orientation(lb1gt_1,147,34). coords(lb1gt_1,72,88,34).  visible(lb1gt_1,34). orientation(lb1gt_2,142,34). coords(lb1gt_2,80,75,34).  visible(lb1gt_2,34).
orientation(lb1gt_1,147,35). coords(lb1gt_1,72,88,35).  visible(lb1gt_1,35). orientation(lb1gt_2,142,35). coords(lb1gt_2,80,75,35).  visible(lb1gt_2,35).
orientation(lb1gt_1,147,36). coords(lb1gt_1,72,88,36).  visible(lb1gt_1,36). orientation(lb1gt_2,142,36). coords(lb1gt_2,80,75,36).  visible(lb1gt_2,36).
orientation(lb1gt_1,147,37). coords(lb1gt_1,72,88,37).  visible(lb1gt_1,37). orientation(lb1gt_2,142,37). coords(lb1gt_2,80,75,37).  visible(lb1gt_2,37).
happensAt(disappear(lb1gt_1),36). happensAt(disappear(lb1gt_2),36).
1{simpleEvent(T,lb1gt_1,active); simpleEvent(T,lb1gt_1,inactive); simpleEvent(T,lb1gt_1,walking); simpleEvent(T,lb1gt_1,running)}1 :- T=12..36.
1{simpleEvent(T,lb1gt_2,active); simpleEvent(T,lb1gt_2,inactive); simpleEvent(T,lb1gt_2,walking); simpleEvent(T,lb1gt_2,running)}1 :- T=12..36.
:- holdsAt(moving(lb1gt_1,lb1gt_2),12). :- holdsAt(moving(lb1gt_2,lb1gt_1),12).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),13). :- not holdsAt(moving(lb1gt_2,lb1gt_1),13).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),14). :- not holdsAt(moving(lb1gt_2,lb1gt_1),14).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),15). :- not holdsAt(moving(lb1gt_2,lb1gt_1),15).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),16). :- not holdsAt(moving(lb1gt_2,lb1gt_1),16).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),17). :- not holdsAt(moving(lb1gt_2,lb1gt_1),17).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),18). :- not holdsAt(moving(lb1gt_2,lb1gt_1),18).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),19). :- not holdsAt(moving(lb1gt_2,lb1gt_1),19).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),20). :- not holdsAt(moving(lb1gt_2,lb1gt_1),20).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),21). :- not holdsAt(moving(lb1gt_2,lb1gt_1),21).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),22). :- not holdsAt(moving(lb1gt_2,lb1gt_1),22).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),23). :- not holdsAt(moving(lb1gt_2,lb1gt_1),23).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),24). :- not holdsAt(moving(lb1gt_2,lb1gt_1),24).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),25). :- not holdsAt(moving(lb1gt_2,lb1gt_1),25).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),26). :- not holdsAt(moving(lb1gt_2,lb1gt_1),26).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),27). :- not holdsAt(moving(lb1gt_2,lb1gt_1),27).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),28). :- not holdsAt(moving(lb1gt_2,lb1gt_1),28).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),29). :- not holdsAt(moving(lb1gt_2,lb1gt_1),29).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),30). :- not holdsAt(moving(lb1gt_2,lb1gt_1),30).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),31). :- not holdsAt(moving(lb1gt_2,lb1gt_1),31).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),32). :- not holdsAt(moving(lb1gt_2,lb1gt_1),32).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),33). :- not holdsAt(moving(lb1gt_2,lb1gt_1),33).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),34). :- not holdsAt(moving(lb1gt_2,lb1gt_1),34).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),35). :- not holdsAt(moving(lb1gt_2,lb1gt_1),35).
:- not holdsAt(moving(lb1gt_1,lb1gt_2),36). :- not holdsAt(moving(lb1gt_2,lb1gt_1),36).
:- holdsAt(moving(lb1gt_1,lb1gt_2),37). :- holdsAt(moving(lb1gt_2,lb1gt_1),37).
:- simpleEvent(T,P,inactive), T>12 , T<36.
:- simpleEvent(T,P,running), T>12 , T<36.

#show simpleEvent/3.