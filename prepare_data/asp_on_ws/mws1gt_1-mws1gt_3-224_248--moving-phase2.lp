

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

	person(mws1gt_1;mws1gt_3).
orientation(mws1gt_1,155,224). coords(mws1gt_1,125,130,224).  visible(mws1gt_1,224). orientation(mws1gt_3,162,224). coords(mws1gt_3,121,152,224).  visible(mws1gt_3,224).
orientation(mws1gt_1,155,225). coords(mws1gt_1,125,130,225).  visible(mws1gt_1,225). orientation(mws1gt_3,162,225). coords(mws1gt_3,120,151,225).  visible(mws1gt_3,225).
orientation(mws1gt_1,155,226). coords(mws1gt_1,125,130,226).  visible(mws1gt_1,226). orientation(mws1gt_3,162,226). coords(mws1gt_3,118,150,226).  visible(mws1gt_3,226).
orientation(mws1gt_1,155,227). coords(mws1gt_1,124,130,227).  visible(mws1gt_1,227). orientation(mws1gt_3,162,227). coords(mws1gt_3,116,149,227).  visible(mws1gt_3,227).
orientation(mws1gt_1,155,228). coords(mws1gt_1,123,129,228).  visible(mws1gt_1,228). orientation(mws1gt_3,162,228). coords(mws1gt_3,115,149,228).  visible(mws1gt_3,228).
orientation(mws1gt_1,155,229). coords(mws1gt_1,122,129,229).  visible(mws1gt_1,229). orientation(mws1gt_3,162,229). coords(mws1gt_3,113,149,229).  visible(mws1gt_3,229).
orientation(mws1gt_1,155,230). coords(mws1gt_1,121,128,230).  visible(mws1gt_1,230). orientation(mws1gt_3,162,230). coords(mws1gt_3,113,149,230).  visible(mws1gt_3,230).
orientation(mws1gt_1,155,231). coords(mws1gt_1,120,127,231).  visible(mws1gt_1,231). orientation(mws1gt_3,162,231). coords(mws1gt_3,113,149,231).  visible(mws1gt_3,231).
orientation(mws1gt_1,155,232). coords(mws1gt_1,118,127,232).  visible(mws1gt_1,232). orientation(mws1gt_3,162,232). coords(mws1gt_3,113,149,232).  visible(mws1gt_3,232).
orientation(mws1gt_1,155,233). coords(mws1gt_1,118,126,233).  visible(mws1gt_1,233). orientation(mws1gt_3,162,233). coords(mws1gt_3,112,148,233).  visible(mws1gt_3,233).
orientation(mws1gt_1,155,234). coords(mws1gt_1,117,125,234).  visible(mws1gt_1,234). orientation(mws1gt_3,162,234). coords(mws1gt_3,112,148,234).  visible(mws1gt_3,234).
orientation(mws1gt_1,155,235). coords(mws1gt_1,117,125,235).  visible(mws1gt_1,235). orientation(mws1gt_3,162,235). coords(mws1gt_3,112,148,235).  visible(mws1gt_3,235).
orientation(mws1gt_1,155,236). coords(mws1gt_1,117,125,236).  visible(mws1gt_1,236). orientation(mws1gt_3,157,236). coords(mws1gt_3,111,148,236).  visible(mws1gt_3,236).
orientation(mws1gt_1,155,237). coords(mws1gt_1,117,124,237).  visible(mws1gt_1,237). orientation(mws1gt_3,157,237). coords(mws1gt_3,110,147,237).  visible(mws1gt_3,237).
orientation(mws1gt_1,155,238). coords(mws1gt_1,116,123,238).  visible(mws1gt_1,238). orientation(mws1gt_3,157,238). coords(mws1gt_3,109,147,238).  visible(mws1gt_3,238).
orientation(mws1gt_1,155,239). coords(mws1gt_1,116,123,239).  visible(mws1gt_1,239). orientation(mws1gt_3,157,239). coords(mws1gt_3,107,147,239).  visible(mws1gt_3,239).
orientation(mws1gt_1,155,240). coords(mws1gt_1,116,123,240).  visible(mws1gt_1,240). orientation(mws1gt_3,157,240). coords(mws1gt_3,106,146,240).  visible(mws1gt_3,240).
orientation(mws1gt_1,155,241). coords(mws1gt_1,116,123,241).  visible(mws1gt_1,241). orientation(mws1gt_3,157,241). coords(mws1gt_3,104,146,241).  visible(mws1gt_3,241).
orientation(mws1gt_1,155,242). coords(mws1gt_1,115,122,242).  visible(mws1gt_1,242). orientation(mws1gt_3,157,242). coords(mws1gt_3,103,144,242).  visible(mws1gt_3,242).
orientation(mws1gt_1,155,243). coords(mws1gt_1,114,122,243).  visible(mws1gt_1,243). orientation(mws1gt_3,157,243). coords(mws1gt_3,103,143,243).  visible(mws1gt_3,243).
orientation(mws1gt_1,155,244). coords(mws1gt_1,113,122,244).  visible(mws1gt_1,244). orientation(mws1gt_3,157,244). coords(mws1gt_3,103,142,244).  visible(mws1gt_3,244).
orientation(mws1gt_1,155,245). coords(mws1gt_1,112,122,245).  visible(mws1gt_1,245). orientation(mws1gt_3,157,245). coords(mws1gt_3,102,141,245).  visible(mws1gt_3,245).
orientation(mws1gt_1,155,246). coords(mws1gt_1,110,122,246).  visible(mws1gt_1,246). orientation(mws1gt_3,157,246). coords(mws1gt_3,102,141,246).  visible(mws1gt_3,246).
orientation(mws1gt_1,155,247). coords(mws1gt_1,109,121,247).  visible(mws1gt_1,247). orientation(mws1gt_3,157,247). coords(mws1gt_3,102,141,247).  visible(mws1gt_3,247).
orientation(mws1gt_1,155,248). coords(mws1gt_1,109,121,248).  visible(mws1gt_1,248). orientation(mws1gt_3,157,248). coords(mws1gt_3,102,141,248).  visible(mws1gt_3,248).
orientation(mws1gt_1,155,249). coords(mws1gt_1,109,121,249).  visible(mws1gt_1,249). orientation(mws1gt_3,157,249). coords(mws1gt_3,102,141,249).  visible(mws1gt_3,249).
happensAt(disappear(mws1gt_1),248). happensAt(disappear(mws1gt_3),248).
1{simpleEvent(T,mws1gt_1,active); simpleEvent(T,mws1gt_1,inactive); simpleEvent(T,mws1gt_1,walking); simpleEvent(T,mws1gt_1,running)}1 :- T=224..248.
1{simpleEvent(T,mws1gt_3,active); simpleEvent(T,mws1gt_3,inactive); simpleEvent(T,mws1gt_3,walking); simpleEvent(T,mws1gt_3,running)}1 :- T=224..248.
:- holdsAt(moving(mws1gt_1,mws1gt_3),224). :- holdsAt(moving(mws1gt_3,mws1gt_1),224).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),225). :- not holdsAt(moving(mws1gt_3,mws1gt_1),225).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),226). :- not holdsAt(moving(mws1gt_3,mws1gt_1),226).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),227). :- not holdsAt(moving(mws1gt_3,mws1gt_1),227).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),228). :- not holdsAt(moving(mws1gt_3,mws1gt_1),228).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),229). :- not holdsAt(moving(mws1gt_3,mws1gt_1),229).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),230). :- not holdsAt(moving(mws1gt_3,mws1gt_1),230).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),231). :- not holdsAt(moving(mws1gt_3,mws1gt_1),231).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),232). :- not holdsAt(moving(mws1gt_3,mws1gt_1),232).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),233). :- not holdsAt(moving(mws1gt_3,mws1gt_1),233).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),234). :- not holdsAt(moving(mws1gt_3,mws1gt_1),234).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),235). :- not holdsAt(moving(mws1gt_3,mws1gt_1),235).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),236). :- not holdsAt(moving(mws1gt_3,mws1gt_1),236).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),237). :- not holdsAt(moving(mws1gt_3,mws1gt_1),237).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),238). :- not holdsAt(moving(mws1gt_3,mws1gt_1),238).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),239). :- not holdsAt(moving(mws1gt_3,mws1gt_1),239).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),240). :- not holdsAt(moving(mws1gt_3,mws1gt_1),240).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),241). :- not holdsAt(moving(mws1gt_3,mws1gt_1),241).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),242). :- not holdsAt(moving(mws1gt_3,mws1gt_1),242).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),243). :- not holdsAt(moving(mws1gt_3,mws1gt_1),243).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),244). :- not holdsAt(moving(mws1gt_3,mws1gt_1),244).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),245). :- not holdsAt(moving(mws1gt_3,mws1gt_1),245).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),246). :- not holdsAt(moving(mws1gt_3,mws1gt_1),246).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),247). :- not holdsAt(moving(mws1gt_3,mws1gt_1),247).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),248). :- not holdsAt(moving(mws1gt_3,mws1gt_1),248).
:- holdsAt(moving(mws1gt_1,mws1gt_3),249). :- holdsAt(moving(mws1gt_3,mws1gt_1),249).
:- simpleEvent(T,P,inactive), T>224 , T<248.
:- simpleEvent(T,P,running), T>224 , T<248.

#show simpleEvent/3.