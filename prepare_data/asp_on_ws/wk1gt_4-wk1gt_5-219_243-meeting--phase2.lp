

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
orientation(wk1gt_4,164,219). coords(wk1gt_4,47,181,219).  visible(wk1gt_4,219). orientation(wk1gt_5,162,219). coords(wk1gt_5,50,166,219).  visible(wk1gt_5,219).
orientation(wk1gt_4,166,220). coords(wk1gt_4,47,181,220).  visible(wk1gt_4,220). orientation(wk1gt_5,162,220). coords(wk1gt_5,50,166,220).  visible(wk1gt_5,220).
orientation(wk1gt_4,164,221). coords(wk1gt_4,47,181,221).  visible(wk1gt_4,221). orientation(wk1gt_5,162,221). coords(wk1gt_5,50,166,221).  visible(wk1gt_5,221).
orientation(wk1gt_4,168,222). coords(wk1gt_4,47,181,222).  visible(wk1gt_4,222). orientation(wk1gt_5,162,222). coords(wk1gt_5,50,166,222).  visible(wk1gt_5,222).
orientation(wk1gt_4,163,223). coords(wk1gt_4,47,181,223).  visible(wk1gt_4,223). orientation(wk1gt_5,162,223). coords(wk1gt_5,50,166,223).  visible(wk1gt_5,223).
orientation(wk1gt_4,165,224). coords(wk1gt_4,47,181,224).  visible(wk1gt_4,224). orientation(wk1gt_5,162,224). coords(wk1gt_5,50,166,224).  visible(wk1gt_5,224).
orientation(wk1gt_4,165,225). coords(wk1gt_4,47,181,225).  visible(wk1gt_4,225). orientation(wk1gt_5,162,225). coords(wk1gt_5,50,166,225).  visible(wk1gt_5,225).
orientation(wk1gt_4,166,226). coords(wk1gt_4,47,181,226).  visible(wk1gt_4,226). orientation(wk1gt_5,162,226). coords(wk1gt_5,50,166,226).  visible(wk1gt_5,226).
orientation(wk1gt_4,164,227). coords(wk1gt_4,47,181,227).  visible(wk1gt_4,227). orientation(wk1gt_5,162,227). coords(wk1gt_5,50,166,227).  visible(wk1gt_5,227).
orientation(wk1gt_4,164,228). coords(wk1gt_4,47,181,228).  visible(wk1gt_4,228). orientation(wk1gt_5,162,228). coords(wk1gt_5,50,166,228).  visible(wk1gt_5,228).
orientation(wk1gt_4,163,229). coords(wk1gt_4,47,181,229).  visible(wk1gt_4,229). orientation(wk1gt_5,162,229). coords(wk1gt_5,50,166,229).  visible(wk1gt_5,229).
orientation(wk1gt_4,159,230). coords(wk1gt_4,47,181,230).  visible(wk1gt_4,230). orientation(wk1gt_5,162,230). coords(wk1gt_5,50,166,230).  visible(wk1gt_5,230).
orientation(wk1gt_4,162,231). coords(wk1gt_4,47,181,231).  visible(wk1gt_4,231). orientation(wk1gt_5,162,231). coords(wk1gt_5,50,166,231).  visible(wk1gt_5,231).
orientation(wk1gt_4,160,232). coords(wk1gt_4,47,181,232).  visible(wk1gt_4,232). orientation(wk1gt_5,162,232). coords(wk1gt_5,50,166,232).  visible(wk1gt_5,232).
orientation(wk1gt_4,163,233). coords(wk1gt_4,47,181,233).  visible(wk1gt_4,233). orientation(wk1gt_5,162,233). coords(wk1gt_5,50,166,233).  visible(wk1gt_5,233).
orientation(wk1gt_4,163,234). coords(wk1gt_4,47,181,234).  visible(wk1gt_4,234). orientation(wk1gt_5,162,234). coords(wk1gt_5,50,166,234).  visible(wk1gt_5,234).
orientation(wk1gt_4,166,235). coords(wk1gt_4,47,181,235).  visible(wk1gt_4,235). orientation(wk1gt_5,162,235). coords(wk1gt_5,50,166,235).  visible(wk1gt_5,235).
orientation(wk1gt_4,165,236). coords(wk1gt_4,47,181,236).  visible(wk1gt_4,236). orientation(wk1gt_5,162,236). coords(wk1gt_5,50,166,236).  visible(wk1gt_5,236).
orientation(wk1gt_4,162,237). coords(wk1gt_4,47,181,237).  visible(wk1gt_4,237). orientation(wk1gt_5,162,237). coords(wk1gt_5,50,166,237).  visible(wk1gt_5,237).
orientation(wk1gt_4,165,238). coords(wk1gt_4,47,181,238).  visible(wk1gt_4,238). orientation(wk1gt_5,162,238). coords(wk1gt_5,50,166,238).  visible(wk1gt_5,238).
orientation(wk1gt_4,163,239). coords(wk1gt_4,47,181,239).  visible(wk1gt_4,239). orientation(wk1gt_5,162,239). coords(wk1gt_5,50,166,239).  visible(wk1gt_5,239).
orientation(wk1gt_4,163,240). coords(wk1gt_4,47,181,240).  visible(wk1gt_4,240). orientation(wk1gt_5,162,240). coords(wk1gt_5,50,166,240).  visible(wk1gt_5,240).
orientation(wk1gt_4,161,241). coords(wk1gt_4,47,181,241).  visible(wk1gt_4,241). orientation(wk1gt_5,162,241). coords(wk1gt_5,50,166,241).  visible(wk1gt_5,241).
orientation(wk1gt_4,162,242). coords(wk1gt_4,47,181,242).  visible(wk1gt_4,242). orientation(wk1gt_5,162,242). coords(wk1gt_5,50,166,242).  visible(wk1gt_5,242).
orientation(wk1gt_4,160,243). coords(wk1gt_4,47,181,243).  visible(wk1gt_4,243). orientation(wk1gt_5,162,243). coords(wk1gt_5,50,166,243).  visible(wk1gt_5,243).
orientation(wk1gt_4,160,244). coords(wk1gt_4,47,181,244).  visible(wk1gt_4,244). orientation(wk1gt_5,162,244). coords(wk1gt_5,50,166,244).  visible(wk1gt_5,244).
happensAt(disappear(wk1gt_4),243). happensAt(disappear(wk1gt_5),243).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=219..243.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=219..243.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),219). :- holdsAt(meeting(wk1gt_5,wk1gt_4),219).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),220). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),220).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),221). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),221).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),222). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),222).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),223). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),223).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),224). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),224).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),225). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),225).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),226). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),226).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),227). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),227).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),228). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),228).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),229). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),229).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),230). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),230).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),231). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),231).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),232). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),232).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),233). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),233).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),234). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),234).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),235). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),235).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),236). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),236).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),237). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),237).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),238). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),238).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),239). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),239).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),240). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),240).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),241). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),241).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),242). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),242).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),243). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),243).
:- holdsAt(meeting(wk1gt_4,wk1gt_5),244). :- holdsAt(meeting(wk1gt_5,wk1gt_4),244).
:- simpleEvent(T,P,walking), T>219 , T<243.
:- simpleEvent(T,P,running), T>219 , T<243.

#show simpleEvent/3.