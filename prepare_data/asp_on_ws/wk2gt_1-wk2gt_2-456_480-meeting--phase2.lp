

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
orientation(wk2gt_1,125,470). coords(wk2gt_1,115,53,470).  visible(wk2gt_1,470). orientation(wk2gt_2,141,470). coords(wk2gt_2,98,63,470).  visible(wk2gt_2,470).
orientation(wk2gt_1,125,471). coords(wk2gt_1,115,53,471).  visible(wk2gt_1,471). orientation(wk2gt_2,141,471). coords(wk2gt_2,98,63,471).  visible(wk2gt_2,471).
orientation(wk2gt_1,125,472). coords(wk2gt_1,115,53,472).  visible(wk2gt_1,472). orientation(wk2gt_2,141,472). coords(wk2gt_2,98,63,472).  visible(wk2gt_2,472).
orientation(wk2gt_1,125,473). coords(wk2gt_1,115,53,473).  visible(wk2gt_1,473). orientation(wk2gt_2,141,473). coords(wk2gt_2,98,63,473).  visible(wk2gt_2,473).
orientation(wk2gt_1,125,474). coords(wk2gt_1,115,53,474).  visible(wk2gt_1,474). orientation(wk2gt_2,141,474). coords(wk2gt_2,98,63,474).  visible(wk2gt_2,474).
orientation(wk2gt_1,125,475). coords(wk2gt_1,115,53,475).  visible(wk2gt_1,475). orientation(wk2gt_2,141,475). coords(wk2gt_2,98,63,475).  visible(wk2gt_2,475).
orientation(wk2gt_1,125,476). coords(wk2gt_1,115,53,476).  visible(wk2gt_1,476). orientation(wk2gt_2,141,476). coords(wk2gt_2,98,63,476).  visible(wk2gt_2,476).
orientation(wk2gt_1,125,477). coords(wk2gt_1,115,53,477).  visible(wk2gt_1,477). orientation(wk2gt_2,141,477). coords(wk2gt_2,98,63,477).  visible(wk2gt_2,477).
orientation(wk2gt_1,125,478). coords(wk2gt_1,115,53,478).  visible(wk2gt_1,478). orientation(wk2gt_2,141,478). coords(wk2gt_2,98,63,478).  visible(wk2gt_2,478).
orientation(wk2gt_1,125,479). coords(wk2gt_1,115,53,479).  visible(wk2gt_1,479). orientation(wk2gt_2,141,479). coords(wk2gt_2,98,63,479).  visible(wk2gt_2,479).
orientation(wk2gt_1,125,480). coords(wk2gt_1,115,53,480).  visible(wk2gt_1,480). orientation(wk2gt_2,141,480). coords(wk2gt_2,98,63,480).  visible(wk2gt_2,480).
orientation(wk2gt_1,125,481). coords(wk2gt_1,115,53,481).  visible(wk2gt_1,481). orientation(wk2gt_2,141,481). coords(wk2gt_2,98,63,481).  visible(wk2gt_2,481).
happensAt(disappear(wk2gt_1),480). happensAt(disappear(wk2gt_2),480).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=456..480.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=456..480.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),456). :- holdsAt(meeting(wk2gt_2,wk2gt_1),456).
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
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),469). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),469).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),470). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),470).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),471). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),471).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),472). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),472).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),473). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),473).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),474). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),474).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),475). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),475).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),476). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),476).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),477). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),477).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),478). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),478).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),479). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),479).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),480). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),480).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),481). :- holdsAt(meeting(wk2gt_2,wk2gt_1),481).
:- simpleEvent(T,P,walking), T>456 , T<480.
:- simpleEvent(T,P,running), T>456 , T<480.

#show simpleEvent/3.