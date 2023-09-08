

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
orientation(wk2gt_1,125,444). coords(wk2gt_1,115,53,444).  visible(wk2gt_1,444). orientation(wk2gt_2,141,444). coords(wk2gt_2,98,63,444).  visible(wk2gt_2,444).
orientation(wk2gt_1,125,445). coords(wk2gt_1,115,53,445).  visible(wk2gt_1,445). orientation(wk2gt_2,141,445). coords(wk2gt_2,98,63,445).  visible(wk2gt_2,445).
orientation(wk2gt_1,125,446). coords(wk2gt_1,115,53,446).  visible(wk2gt_1,446). orientation(wk2gt_2,141,446). coords(wk2gt_2,98,63,446).  visible(wk2gt_2,446).
orientation(wk2gt_1,125,447). coords(wk2gt_1,115,53,447).  visible(wk2gt_1,447). orientation(wk2gt_2,141,447). coords(wk2gt_2,98,63,447).  visible(wk2gt_2,447).
orientation(wk2gt_1,125,448). coords(wk2gt_1,115,53,448).  visible(wk2gt_1,448). orientation(wk2gt_2,141,448). coords(wk2gt_2,98,63,448).  visible(wk2gt_2,448).
orientation(wk2gt_1,125,449). coords(wk2gt_1,115,53,449).  visible(wk2gt_1,449). orientation(wk2gt_2,141,449). coords(wk2gt_2,98,63,449).  visible(wk2gt_2,449).
orientation(wk2gt_1,125,450). coords(wk2gt_1,115,53,450).  visible(wk2gt_1,450). orientation(wk2gt_2,141,450). coords(wk2gt_2,98,63,450).  visible(wk2gt_2,450).
orientation(wk2gt_1,125,451). coords(wk2gt_1,115,53,451).  visible(wk2gt_1,451). orientation(wk2gt_2,141,451). coords(wk2gt_2,98,63,451).  visible(wk2gt_2,451).
orientation(wk2gt_1,125,452). coords(wk2gt_1,115,53,452).  visible(wk2gt_1,452). orientation(wk2gt_2,141,452). coords(wk2gt_2,98,63,452).  visible(wk2gt_2,452).
orientation(wk2gt_1,125,453). coords(wk2gt_1,115,53,453).  visible(wk2gt_1,453). orientation(wk2gt_2,141,453). coords(wk2gt_2,98,63,453).  visible(wk2gt_2,453).
orientation(wk2gt_1,125,454). coords(wk2gt_1,115,53,454).  visible(wk2gt_1,454). orientation(wk2gt_2,141,454). coords(wk2gt_2,98,63,454).  visible(wk2gt_2,454).
orientation(wk2gt_1,125,455). coords(wk2gt_1,115,53,455).  visible(wk2gt_1,455). orientation(wk2gt_2,141,455). coords(wk2gt_2,98,63,455).  visible(wk2gt_2,455).
orientation(wk2gt_1,125,456). coords(wk2gt_1,115,53,456).  visible(wk2gt_1,456). orientation(wk2gt_2,141,456). coords(wk2gt_2,98,63,456).  visible(wk2gt_2,456).
orientation(wk2gt_1,125,457). coords(wk2gt_1,115,53,457).  visible(wk2gt_1,457). orientation(wk2gt_2,141,457). coords(wk2gt_2,98,63,457).  visible(wk2gt_2,457).
orientation(wk2gt_1,125,458). coords(wk2gt_1,115,53,458).  visible(wk2gt_1,458). orientation(wk2gt_2,141,458). coords(wk2gt_2,98,63,458).  visible(wk2gt_2,458).
orientation(wk2gt_1,125,459). coords(wk2gt_1,115,53,459).  visible(wk2gt_1,459). orientation(wk2gt_2,141,459). coords(wk2gt_2,98,63,459).  visible(wk2gt_2,459).
orientation(wk2gt_1,125,460). coords(wk2gt_1,115,53,460).  visible(wk2gt_1,460). orientation(wk2gt_2,141,460). coords(wk2gt_2,98,63,460).  visible(wk2gt_2,460).
orientation(wk2gt_1,125,461). coords(wk2gt_1,115,53,461).  visible(wk2gt_1,461). orientation(wk2gt_2,141,461). coords(wk2gt_2,98,63,461).  visible(wk2gt_2,461).
orientation(wk2gt_1,125,462). coords(wk2gt_1,115,53,462).  visible(wk2gt_1,462). orientation(wk2gt_2,141,462). coords(wk2gt_2,98,63,462).  visible(wk2gt_2,462).
orientation(wk2gt_1,125,463). coords(wk2gt_1,115,53,463).  visible(wk2gt_1,463). orientation(wk2gt_2,141,463). coords(wk2gt_2,98,63,463).  visible(wk2gt_2,463).
orientation(wk2gt_1,125,464). coords(wk2gt_1,115,53,464).  visible(wk2gt_1,464). orientation(wk2gt_2,141,464). coords(wk2gt_2,98,63,464).  visible(wk2gt_2,464).
orientation(wk2gt_1,125,465). coords(wk2gt_1,115,53,465).  visible(wk2gt_1,465). orientation(wk2gt_2,141,465). coords(wk2gt_2,98,63,465).  visible(wk2gt_2,465).
orientation(wk2gt_1,125,466). coords(wk2gt_1,115,53,466).  visible(wk2gt_1,466). orientation(wk2gt_2,141,466). coords(wk2gt_2,98,63,466).  visible(wk2gt_2,466).
orientation(wk2gt_1,125,467). coords(wk2gt_1,115,53,467).  visible(wk2gt_1,467). orientation(wk2gt_2,141,467). coords(wk2gt_2,98,63,467).  visible(wk2gt_2,467).
orientation(wk2gt_1,125,468). coords(wk2gt_1,115,53,468).  visible(wk2gt_1,468). orientation(wk2gt_2,141,468). coords(wk2gt_2,98,63,468).  visible(wk2gt_2,468).
orientation(wk2gt_1,125,469). coords(wk2gt_1,115,53,469).  visible(wk2gt_1,469). orientation(wk2gt_2,141,469). coords(wk2gt_2,98,63,469).  visible(wk2gt_2,469).
happensAt(disappear(wk2gt_1),468). happensAt(disappear(wk2gt_2),468).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=444..468.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=444..468.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),444). :- holdsAt(meeting(wk2gt_2,wk2gt_1),444).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),445). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),445).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),446). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),446).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),447). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),447).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),448). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),448).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),449). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),449).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),450). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),450).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),451). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),451).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),452). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),452).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),453). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),453).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),454). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),454).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),455). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),455).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),456). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),456).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),457). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),457).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),458). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),458).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),459). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),459).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),460). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),460).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),461). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),461).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),462). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),462).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),463). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),463).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),464). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),464).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),465). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),465).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),466). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),466).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),467). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),467).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),468). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),468).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),469). :- holdsAt(meeting(wk2gt_2,wk2gt_1),469).
:- simpleEvent(T,P,walking), T>444 , T<468.
:- simpleEvent(T,P,running), T>444 , T<468.

#show simpleEvent/3.