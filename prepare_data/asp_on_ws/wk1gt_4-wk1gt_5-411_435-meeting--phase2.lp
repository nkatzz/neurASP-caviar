

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

	person(wk1gt_4;wk1gt_5).
orientation(wk1gt_4,168,411). coords(wk1gt_4,52,176,411).  visible(wk1gt_4,411). orientation(wk1gt_5,162,411). coords(wk1gt_5,50,163,411).  visible(wk1gt_5,411).
orientation(wk1gt_4,162,412). coords(wk1gt_4,52,176,412).  visible(wk1gt_4,412). orientation(wk1gt_5,162,412). coords(wk1gt_5,50,163,412).  visible(wk1gt_5,412).
orientation(wk1gt_4,162,413). coords(wk1gt_4,52,176,413).  visible(wk1gt_4,413). orientation(wk1gt_5,162,413). coords(wk1gt_5,50,163,413).  visible(wk1gt_5,413).
orientation(wk1gt_4,161,414). coords(wk1gt_4,52,176,414).  visible(wk1gt_4,414). orientation(wk1gt_5,162,414). coords(wk1gt_5,50,163,414).  visible(wk1gt_5,414).
orientation(wk1gt_4,163,415). coords(wk1gt_4,52,176,415).  visible(wk1gt_4,415). orientation(wk1gt_5,162,415). coords(wk1gt_5,50,163,415).  visible(wk1gt_5,415).
orientation(wk1gt_4,163,416). coords(wk1gt_4,53,176,416).  visible(wk1gt_4,416). orientation(wk1gt_5,162,416). coords(wk1gt_5,50,163,416).  visible(wk1gt_5,416).
orientation(wk1gt_4,160,417). coords(wk1gt_4,53,175,417).  visible(wk1gt_4,417). orientation(wk1gt_5,162,417). coords(wk1gt_5,50,163,417).  visible(wk1gt_5,417).
orientation(wk1gt_4,159,418). coords(wk1gt_4,53,175,418).  visible(wk1gt_4,418). orientation(wk1gt_5,162,418). coords(wk1gt_5,50,163,418).  visible(wk1gt_5,418).
orientation(wk1gt_4,157,419). coords(wk1gt_4,54,175,419).  visible(wk1gt_4,419). orientation(wk1gt_5,162,419). coords(wk1gt_5,50,163,419).  visible(wk1gt_5,419).
orientation(wk1gt_4,157,420). coords(wk1gt_4,54,175,420).  visible(wk1gt_4,420). orientation(wk1gt_5,162,420). coords(wk1gt_5,50,163,420).  visible(wk1gt_5,420).
orientation(wk1gt_4,157,421). coords(wk1gt_4,54,175,421).  visible(wk1gt_4,421). orientation(wk1gt_5,162,421). coords(wk1gt_5,50,163,421).  visible(wk1gt_5,421).
orientation(wk1gt_4,157,422). coords(wk1gt_4,54,174,422).  visible(wk1gt_4,422). orientation(wk1gt_5,162,422). coords(wk1gt_5,50,163,422).  visible(wk1gt_5,422).
orientation(wk1gt_4,157,423). coords(wk1gt_4,54,174,423).  visible(wk1gt_4,423). orientation(wk1gt_5,162,423). coords(wk1gt_5,50,163,423).  visible(wk1gt_5,423).
orientation(wk1gt_4,157,424). coords(wk1gt_4,54,174,424).  visible(wk1gt_4,424). orientation(wk1gt_5,162,424). coords(wk1gt_5,50,163,424).  visible(wk1gt_5,424).
orientation(wk1gt_4,157,425). coords(wk1gt_4,54,173,425).  visible(wk1gt_4,425). orientation(wk1gt_5,162,425). coords(wk1gt_5,50,163,425).  visible(wk1gt_5,425).
orientation(wk1gt_4,147,426). coords(wk1gt_4,54,173,426).  visible(wk1gt_4,426). orientation(wk1gt_5,162,426). coords(wk1gt_5,50,163,426).  visible(wk1gt_5,426).
orientation(wk1gt_4,147,427). coords(wk1gt_4,54,172,427).  visible(wk1gt_4,427). orientation(wk1gt_5,162,427). coords(wk1gt_5,50,163,427).  visible(wk1gt_5,427).
orientation(wk1gt_4,147,428). coords(wk1gt_4,54,172,428).  visible(wk1gt_4,428). orientation(wk1gt_5,162,428). coords(wk1gt_5,50,163,428).  visible(wk1gt_5,428).
orientation(wk1gt_4,147,429). coords(wk1gt_4,54,172,429).  visible(wk1gt_4,429). orientation(wk1gt_5,162,429). coords(wk1gt_5,50,163,429).  visible(wk1gt_5,429).
orientation(wk1gt_4,147,430). coords(wk1gt_4,54,171,430).  visible(wk1gt_4,430). orientation(wk1gt_5,162,430). coords(wk1gt_5,50,163,430).  visible(wk1gt_5,430).
orientation(wk1gt_4,147,431). coords(wk1gt_4,54,171,431).  visible(wk1gt_4,431). orientation(wk1gt_5,162,431). coords(wk1gt_5,50,163,431).  visible(wk1gt_5,431).
orientation(wk1gt_4,147,432). coords(wk1gt_4,54,171,432).  visible(wk1gt_4,432). orientation(wk1gt_5,162,432). coords(wk1gt_5,50,163,432).  visible(wk1gt_5,432).
orientation(wk1gt_4,147,433). coords(wk1gt_4,54,171,433).  visible(wk1gt_4,433). orientation(wk1gt_5,162,433). coords(wk1gt_5,50,163,433).  visible(wk1gt_5,433).
orientation(wk1gt_4,147,434). coords(wk1gt_4,54,170,434).  visible(wk1gt_4,434). orientation(wk1gt_5,162,434). coords(wk1gt_5,51,162,434).  visible(wk1gt_5,434).
orientation(wk1gt_4,147,435). coords(wk1gt_4,55,170,435).  visible(wk1gt_4,435). orientation(wk1gt_5,162,435). coords(wk1gt_5,51,162,435).  visible(wk1gt_5,435).
orientation(wk1gt_4,147,436). coords(wk1gt_4,55,170,436).  visible(wk1gt_4,436). orientation(wk1gt_5,162,436). coords(wk1gt_5,51,162,436).  visible(wk1gt_5,436).
happensAt(disappear(wk1gt_4),435). happensAt(disappear(wk1gt_5),435).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=411..435.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=411..435.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),411). :- holdsAt(meeting(wk1gt_5,wk1gt_4),411).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),412). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),412).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),413). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),413).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),414). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),414).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),415). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),415).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),416). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),416).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),417). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),417).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),418). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),418).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),419). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),419).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),420). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),420).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),421). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),421).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),422). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),422).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),423). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),423).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),424). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),424).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),425). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),425).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),426). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),426).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),427). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),427).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),428). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),428).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),429). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),429).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),430). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),430).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),431). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),431).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),432). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),432).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),433). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),433).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),434). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),434).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),435). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),435).
:- holdsAt(meeting(wk1gt_4,wk1gt_5),436). :- holdsAt(meeting(wk1gt_5,wk1gt_4),436).
:- simpleEvent(T,P,walking), T>411 , T<435.
:- simpleEvent(T,P,running), T>411 , T<435.

#show simpleEvent/3.