

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
orientation(lb2gt_2,147,25). coords(lb2gt_2,20,51,25).  visible(lb2gt_2,25). orientation(lb2gt_3,146,25). coords(lb2gt_3,22,35,25).  visible(lb2gt_3,25).
orientation(lb2gt_2,147,26). coords(lb2gt_2,21,52,26).  visible(lb2gt_2,26). orientation(lb2gt_3,146,26). coords(lb2gt_3,22,35,26).  visible(lb2gt_3,26).
orientation(lb2gt_2,147,27). coords(lb2gt_2,21,52,27).  visible(lb2gt_2,27). orientation(lb2gt_3,146,27). coords(lb2gt_3,22,35,27).  visible(lb2gt_3,27).
orientation(lb2gt_2,147,28). coords(lb2gt_2,21,52,28).  visible(lb2gt_2,28). orientation(lb2gt_3,146,28). coords(lb2gt_3,22,37,28).  visible(lb2gt_3,28).
orientation(lb2gt_2,147,29). coords(lb2gt_2,20,52,29).  visible(lb2gt_2,29). orientation(lb2gt_3,146,29). coords(lb2gt_3,23,37,29).  visible(lb2gt_3,29).
orientation(lb2gt_2,147,30). coords(lb2gt_2,20,52,30).  visible(lb2gt_2,30). orientation(lb2gt_3,146,30). coords(lb2gt_3,24,38,30).  visible(lb2gt_3,30).
orientation(lb2gt_2,147,31). coords(lb2gt_2,20,53,31).  visible(lb2gt_2,31). orientation(lb2gt_3,146,31). coords(lb2gt_3,24,39,31).  visible(lb2gt_3,31).
orientation(lb2gt_2,147,32). coords(lb2gt_2,20,53,32).  visible(lb2gt_2,32). orientation(lb2gt_3,146,32). coords(lb2gt_3,24,39,32).  visible(lb2gt_3,32).
orientation(lb2gt_2,147,33). coords(lb2gt_2,20,53,33).  visible(lb2gt_2,33). orientation(lb2gt_3,146,33). coords(lb2gt_3,24,39,33).  visible(lb2gt_3,33).
orientation(lb2gt_2,147,34). coords(lb2gt_2,20,53,34).  visible(lb2gt_2,34). orientation(lb2gt_3,146,34). coords(lb2gt_3,24,39,34).  visible(lb2gt_3,34).
orientation(lb2gt_2,147,35). coords(lb2gt_2,20,53,35).  visible(lb2gt_2,35). orientation(lb2gt_3,146,35). coords(lb2gt_3,23,40,35).  visible(lb2gt_3,35).
orientation(lb2gt_2,147,36). coords(lb2gt_2,20,53,36).  visible(lb2gt_2,36). orientation(lb2gt_3,146,36). coords(lb2gt_3,23,40,36).  visible(lb2gt_3,36).
orientation(lb2gt_2,147,37). coords(lb2gt_2,20,53,37).  visible(lb2gt_2,37). orientation(lb2gt_3,146,37). coords(lb2gt_3,23,40,37).  visible(lb2gt_3,37).
happensAt(disappear(lb2gt_2),36). happensAt(disappear(lb2gt_3),36).
1{simpleEvent(T,lb2gt_2,active); simpleEvent(T,lb2gt_2,inactive); simpleEvent(T,lb2gt_2,walking); simpleEvent(T,lb2gt_2,running)}1 :- T=12..36.
1{simpleEvent(T,lb2gt_3,active); simpleEvent(T,lb2gt_3,inactive); simpleEvent(T,lb2gt_3,walking); simpleEvent(T,lb2gt_3,running)}1 :- T=12..36.
:- holdsAt(moving(lb2gt_2,lb2gt_3),12). :- holdsAt(moving(lb2gt_3,lb2gt_2),12).
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
:- not holdsAt(moving(lb2gt_2,lb2gt_3),25). :- not holdsAt(moving(lb2gt_3,lb2gt_2),25).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),26). :- not holdsAt(moving(lb2gt_3,lb2gt_2),26).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),27). :- not holdsAt(moving(lb2gt_3,lb2gt_2),27).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),28). :- not holdsAt(moving(lb2gt_3,lb2gt_2),28).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),29). :- not holdsAt(moving(lb2gt_3,lb2gt_2),29).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),30). :- not holdsAt(moving(lb2gt_3,lb2gt_2),30).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),31). :- not holdsAt(moving(lb2gt_3,lb2gt_2),31).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),32). :- not holdsAt(moving(lb2gt_3,lb2gt_2),32).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),33). :- not holdsAt(moving(lb2gt_3,lb2gt_2),33).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),34). :- not holdsAt(moving(lb2gt_3,lb2gt_2),34).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),35). :- not holdsAt(moving(lb2gt_3,lb2gt_2),35).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),36). :- not holdsAt(moving(lb2gt_3,lb2gt_2),36).
:- holdsAt(moving(lb2gt_2,lb2gt_3),37). :- holdsAt(moving(lb2gt_3,lb2gt_2),37).
:- simpleEvent(T,P,inactive), T>12 , T<36.
:- simpleEvent(T,P,running), T>12 , T<36.

#show simpleEvent/3.