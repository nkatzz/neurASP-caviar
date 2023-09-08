

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

	person(spgt_0;spgt_1).
orientation(spgt_0,148,216). coords(spgt_0,113,114,216).  visible(spgt_0,216). orientation(spgt_1,148,216). coords(spgt_1,108,103,216).  visible(spgt_1,216).
orientation(spgt_0,148,217). coords(spgt_0,113,114,217).  visible(spgt_0,217). orientation(spgt_1,148,217). coords(spgt_1,108,103,217).  visible(spgt_1,217).
orientation(spgt_0,148,218). coords(spgt_0,113,114,218).  visible(spgt_0,218). orientation(spgt_1,148,218). coords(spgt_1,108,103,218).  visible(spgt_1,218).
orientation(spgt_0,148,219). coords(spgt_0,113,114,219).  visible(spgt_0,219). orientation(spgt_1,148,219). coords(spgt_1,108,103,219).  visible(spgt_1,219).
orientation(spgt_0,148,220). coords(spgt_0,113,114,220).  visible(spgt_0,220). orientation(spgt_1,148,220). coords(spgt_1,108,103,220).  visible(spgt_1,220).
orientation(spgt_0,148,221). coords(spgt_0,113,114,221).  visible(spgt_0,221). orientation(spgt_1,148,221). coords(spgt_1,108,103,221).  visible(spgt_1,221).
orientation(spgt_0,148,222). coords(spgt_0,113,114,222).  visible(spgt_0,222). orientation(spgt_1,148,222). coords(spgt_1,108,103,222).  visible(spgt_1,222).
orientation(spgt_0,148,223). coords(spgt_0,113,114,223).  visible(spgt_0,223). orientation(spgt_1,148,223). coords(spgt_1,108,103,223).  visible(spgt_1,223).
orientation(spgt_0,148,224). coords(spgt_0,113,114,224).  visible(spgt_0,224). orientation(spgt_1,148,224). coords(spgt_1,109,103,224).  visible(spgt_1,224).
orientation(spgt_0,148,225). coords(spgt_0,113,114,225).  visible(spgt_0,225). orientation(spgt_1,148,225). coords(spgt_1,109,102,225).  visible(spgt_1,225).
orientation(spgt_0,148,226). coords(spgt_0,113,114,226).  visible(spgt_0,226). orientation(spgt_1,148,226). coords(spgt_1,109,102,226).  visible(spgt_1,226).
orientation(spgt_0,148,227). coords(spgt_0,113,114,227).  visible(spgt_0,227). orientation(spgt_1,148,227). coords(spgt_1,109,102,227).  visible(spgt_1,227).
orientation(spgt_0,148,228). coords(spgt_0,113,114,228).  visible(spgt_0,228). orientation(spgt_1,148,228). coords(spgt_1,109,101,228).  visible(spgt_1,228).
orientation(spgt_0,148,229). coords(spgt_0,113,114,229).  visible(spgt_0,229). orientation(spgt_1,148,229). coords(spgt_1,109,101,229).  visible(spgt_1,229).
orientation(spgt_0,148,230). coords(spgt_0,113,114,230).  visible(spgt_0,230). orientation(spgt_1,148,230). coords(spgt_1,109,100,230).  visible(spgt_1,230).
orientation(spgt_0,148,231). coords(spgt_0,113,114,231).  visible(spgt_0,231). orientation(spgt_1,148,231). coords(spgt_1,110,100,231).  visible(spgt_1,231).
orientation(spgt_0,148,232). coords(spgt_0,113,114,232).  visible(spgt_0,232). orientation(spgt_1,147,232). coords(spgt_1,110,100,232).  visible(spgt_1,232).
orientation(spgt_0,148,233). coords(spgt_0,114,114,233).  visible(spgt_0,233). orientation(spgt_1,144,233). coords(spgt_1,110,99,233).  visible(spgt_1,233).
orientation(spgt_0,148,234). coords(spgt_0,114,114,234).  visible(spgt_0,234). orientation(spgt_1,141,234). coords(spgt_1,110,99,234).  visible(spgt_1,234).
orientation(spgt_0,148,235). coords(spgt_0,114,114,235).  visible(spgt_0,235). orientation(spgt_1,141,235). coords(spgt_1,110,99,235).  visible(spgt_1,235).
orientation(spgt_0,148,236). coords(spgt_0,115,113,236).  visible(spgt_0,236). orientation(spgt_1,141,236). coords(spgt_1,111,98,236).  visible(spgt_1,236).
orientation(spgt_0,148,237). coords(spgt_0,115,113,237).  visible(spgt_0,237). orientation(spgt_1,141,237). coords(spgt_1,112,98,237).  visible(spgt_1,237).
orientation(spgt_0,148,238). coords(spgt_0,116,113,238).  visible(spgt_0,238). orientation(spgt_1,141,238). coords(spgt_1,113,98,238).  visible(spgt_1,238).
orientation(spgt_0,148,239). coords(spgt_0,116,113,239).  visible(spgt_0,239). orientation(spgt_1,141,239). coords(spgt_1,114,97,239).  visible(spgt_1,239).
orientation(spgt_0,148,240). coords(spgt_0,116,112,240).  visible(spgt_0,240). orientation(spgt_1,141,240). coords(spgt_1,115,96,240).  visible(spgt_1,240).
orientation(spgt_0,148,241). coords(spgt_0,116,112,241).  visible(spgt_0,241). orientation(spgt_1,141,241). coords(spgt_1,115,96,241).  visible(spgt_1,241).
happensAt(disappear(spgt_0),240). happensAt(disappear(spgt_1),240).
1{simpleEvent(T,spgt_0,active); simpleEvent(T,spgt_0,inactive); simpleEvent(T,spgt_0,walking); simpleEvent(T,spgt_0,running)}1 :- T=216..240.
1{simpleEvent(T,spgt_1,active); simpleEvent(T,spgt_1,inactive); simpleEvent(T,spgt_1,walking); simpleEvent(T,spgt_1,running)}1 :- T=216..240.
:- holdsAt(meeting(spgt_0,spgt_1),216). :- holdsAt(meeting(spgt_1,spgt_0),216).
:- not holdsAt(meeting(spgt_0,spgt_1),217). :- not holdsAt(meeting(spgt_1,spgt_0),217).
:- not holdsAt(meeting(spgt_0,spgt_1),218). :- not holdsAt(meeting(spgt_1,spgt_0),218).
:- not holdsAt(meeting(spgt_0,spgt_1),219). :- not holdsAt(meeting(spgt_1,spgt_0),219).
:- not holdsAt(meeting(spgt_0,spgt_1),220). :- not holdsAt(meeting(spgt_1,spgt_0),220).
:- not holdsAt(meeting(spgt_0,spgt_1),221). :- not holdsAt(meeting(spgt_1,spgt_0),221).
:- not holdsAt(meeting(spgt_0,spgt_1),222). :- not holdsAt(meeting(spgt_1,spgt_0),222).
:- not holdsAt(meeting(spgt_0,spgt_1),223). :- not holdsAt(meeting(spgt_1,spgt_0),223).
:- not holdsAt(meeting(spgt_0,spgt_1),224). :- not holdsAt(meeting(spgt_1,spgt_0),224).
:- not holdsAt(meeting(spgt_0,spgt_1),225). :- not holdsAt(meeting(spgt_1,spgt_0),225).
:- not holdsAt(meeting(spgt_0,spgt_1),226). :- not holdsAt(meeting(spgt_1,spgt_0),226).
:- not holdsAt(meeting(spgt_0,spgt_1),227). :- not holdsAt(meeting(spgt_1,spgt_0),227).
:- not holdsAt(meeting(spgt_0,spgt_1),228). :- not holdsAt(meeting(spgt_1,spgt_0),228).
:- not holdsAt(meeting(spgt_0,spgt_1),229). :- not holdsAt(meeting(spgt_1,spgt_0),229).
:- not holdsAt(meeting(spgt_0,spgt_1),230). :- not holdsAt(meeting(spgt_1,spgt_0),230).
:- not holdsAt(meeting(spgt_0,spgt_1),231). :- not holdsAt(meeting(spgt_1,spgt_0),231).
:- not holdsAt(meeting(spgt_0,spgt_1),232). :- not holdsAt(meeting(spgt_1,spgt_0),232).
:- not holdsAt(meeting(spgt_0,spgt_1),233). :- not holdsAt(meeting(spgt_1,spgt_0),233).
:- not holdsAt(meeting(spgt_0,spgt_1),234). :- not holdsAt(meeting(spgt_1,spgt_0),234).
:- not holdsAt(meeting(spgt_0,spgt_1),235). :- not holdsAt(meeting(spgt_1,spgt_0),235).
:- not holdsAt(meeting(spgt_0,spgt_1),236). :- not holdsAt(meeting(spgt_1,spgt_0),236).
:- not holdsAt(meeting(spgt_0,spgt_1),237). :- not holdsAt(meeting(spgt_1,spgt_0),237).
:- not holdsAt(meeting(spgt_0,spgt_1),238). :- not holdsAt(meeting(spgt_1,spgt_0),238).
:- not holdsAt(meeting(spgt_0,spgt_1),239). :- not holdsAt(meeting(spgt_1,spgt_0),239).
:- not holdsAt(meeting(spgt_0,spgt_1),240). :- not holdsAt(meeting(spgt_1,spgt_0),240).
:- holdsAt(meeting(spgt_0,spgt_1),241). :- holdsAt(meeting(spgt_1,spgt_0),241).
:- simpleEvent(T,P,walking), T>216 , T<240.
:- simpleEvent(T,P,running), T>216 , T<240.

#show simpleEvent/3.