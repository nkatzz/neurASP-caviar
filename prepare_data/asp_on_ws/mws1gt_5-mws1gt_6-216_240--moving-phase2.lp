

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

	person(mws1gt_5;mws1gt_6).
orientation(mws1gt_5,161,216). coords(mws1gt_5,46,117,216).  visible(mws1gt_5,216). orientation(mws1gt_6,146,216). coords(mws1gt_6,55,96,216).  visible(mws1gt_6,216).
orientation(mws1gt_5,161,217). coords(mws1gt_5,46,117,217).  visible(mws1gt_5,217). orientation(mws1gt_6,146,217). coords(mws1gt_6,55,96,217).  visible(mws1gt_6,217).
orientation(mws1gt_5,161,218). coords(mws1gt_5,46,118,218).  visible(mws1gt_5,218). orientation(mws1gt_6,146,218). coords(mws1gt_6,55,97,218).  visible(mws1gt_6,218).
orientation(mws1gt_5,161,219). coords(mws1gt_5,46,118,219).  visible(mws1gt_5,219). orientation(mws1gt_6,146,219). coords(mws1gt_6,55,97,219).  visible(mws1gt_6,219).
orientation(mws1gt_5,161,220). coords(mws1gt_5,46,119,220).  visible(mws1gt_5,220). orientation(mws1gt_6,146,220). coords(mws1gt_6,55,97,220).  visible(mws1gt_6,220).
orientation(mws1gt_5,161,221). coords(mws1gt_5,46,120,221).  visible(mws1gt_5,221). orientation(mws1gt_6,146,221). coords(mws1gt_6,55,98,221).  visible(mws1gt_6,221).
orientation(mws1gt_5,161,222). coords(mws1gt_5,46,120,222).  visible(mws1gt_5,222). orientation(mws1gt_6,146,222). coords(mws1gt_6,55,98,222).  visible(mws1gt_6,222).
orientation(mws1gt_5,161,223). coords(mws1gt_5,46,120,223).  visible(mws1gt_5,223). orientation(mws1gt_6,146,223). coords(mws1gt_6,55,99,223).  visible(mws1gt_6,223).
orientation(mws1gt_5,161,224). coords(mws1gt_5,46,120,224).  visible(mws1gt_5,224). orientation(mws1gt_6,146,224). coords(mws1gt_6,54,99,224).  visible(mws1gt_6,224).
orientation(mws1gt_5,161,225). coords(mws1gt_5,46,120,225).  visible(mws1gt_5,225). orientation(mws1gt_6,146,225). coords(mws1gt_6,53,100,225).  visible(mws1gt_6,225).
orientation(mws1gt_5,161,226). coords(mws1gt_5,45,120,226).  visible(mws1gt_5,226). orientation(mws1gt_6,146,226). coords(mws1gt_6,53,100,226).  visible(mws1gt_6,226).
orientation(mws1gt_5,161,227). coords(mws1gt_5,45,121,227).  visible(mws1gt_5,227). orientation(mws1gt_6,146,227). coords(mws1gt_6,52,101,227).  visible(mws1gt_6,227).
orientation(mws1gt_5,161,228). coords(mws1gt_5,45,124,228).  visible(mws1gt_5,228). orientation(mws1gt_6,146,228). coords(mws1gt_6,52,101,228).  visible(mws1gt_6,228).
orientation(mws1gt_5,161,229). coords(mws1gt_5,45,126,229).  visible(mws1gt_5,229). orientation(mws1gt_6,146,229). coords(mws1gt_6,52,102,229).  visible(mws1gt_6,229).
orientation(mws1gt_5,161,230). coords(mws1gt_5,45,127,230).  visible(mws1gt_5,230). orientation(mws1gt_6,146,230). coords(mws1gt_6,48,102,230).  visible(mws1gt_6,230).
orientation(mws1gt_5,161,231). coords(mws1gt_5,45,128,231).  visible(mws1gt_5,231). orientation(mws1gt_6,146,231). coords(mws1gt_6,48,103,231).  visible(mws1gt_6,231).
orientation(mws1gt_5,161,232). coords(mws1gt_5,45,128,232).  visible(mws1gt_5,232). orientation(mws1gt_6,146,232). coords(mws1gt_6,48,103,232).  visible(mws1gt_6,232).
orientation(mws1gt_5,161,233). coords(mws1gt_5,45,129,233).  visible(mws1gt_5,233). orientation(mws1gt_6,146,233). coords(mws1gt_6,48,103,233).  visible(mws1gt_6,233).
orientation(mws1gt_5,161,234). coords(mws1gt_5,45,129,234).  visible(mws1gt_5,234). orientation(mws1gt_6,146,234). coords(mws1gt_6,48,105,234).  visible(mws1gt_6,234).
orientation(mws1gt_5,161,235). coords(mws1gt_5,45,130,235).  visible(mws1gt_5,235). orientation(mws1gt_6,146,235). coords(mws1gt_6,48,106,235).  visible(mws1gt_6,235).
orientation(mws1gt_5,161,236). coords(mws1gt_5,44,131,236).  visible(mws1gt_5,236). orientation(mws1gt_6,146,236). coords(mws1gt_6,48,106,236).  visible(mws1gt_6,236).
orientation(mws1gt_5,161,237). coords(mws1gt_5,43,132,237).  visible(mws1gt_5,237). orientation(mws1gt_6,146,237). coords(mws1gt_6,44,107,237).  visible(mws1gt_6,237).
orientation(mws1gt_5,161,238). coords(mws1gt_5,43,133,238).  visible(mws1gt_5,238). orientation(mws1gt_6,146,238). coords(mws1gt_6,44,108,238).  visible(mws1gt_6,238).
orientation(mws1gt_5,161,239). coords(mws1gt_5,43,134,239).  visible(mws1gt_5,239). orientation(mws1gt_6,146,239). coords(mws1gt_6,44,109,239).  visible(mws1gt_6,239).
orientation(mws1gt_5,161,240). coords(mws1gt_5,43,134,240).  visible(mws1gt_5,240). orientation(mws1gt_6,146,240). coords(mws1gt_6,44,110,240).  visible(mws1gt_6,240).
orientation(mws1gt_5,161,241). coords(mws1gt_5,43,134,241).  visible(mws1gt_5,241). orientation(mws1gt_6,146,241). coords(mws1gt_6,44,110,241).  visible(mws1gt_6,241).
happensAt(disappear(mws1gt_5),240). happensAt(disappear(mws1gt_6),240).
1{simpleEvent(T,mws1gt_5,active); simpleEvent(T,mws1gt_5,inactive); simpleEvent(T,mws1gt_5,walking); simpleEvent(T,mws1gt_5,running)}1 :- T=216..240.
1{simpleEvent(T,mws1gt_6,active); simpleEvent(T,mws1gt_6,inactive); simpleEvent(T,mws1gt_6,walking); simpleEvent(T,mws1gt_6,running)}1 :- T=216..240.
:- holdsAt(moving(mws1gt_5,mws1gt_6),216). :- holdsAt(moving(mws1gt_6,mws1gt_5),216).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),217). :- not holdsAt(moving(mws1gt_6,mws1gt_5),217).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),218). :- not holdsAt(moving(mws1gt_6,mws1gt_5),218).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),219). :- not holdsAt(moving(mws1gt_6,mws1gt_5),219).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),220). :- not holdsAt(moving(mws1gt_6,mws1gt_5),220).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),221). :- not holdsAt(moving(mws1gt_6,mws1gt_5),221).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),222). :- not holdsAt(moving(mws1gt_6,mws1gt_5),222).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),223). :- not holdsAt(moving(mws1gt_6,mws1gt_5),223).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),224). :- not holdsAt(moving(mws1gt_6,mws1gt_5),224).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),225). :- not holdsAt(moving(mws1gt_6,mws1gt_5),225).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),226). :- not holdsAt(moving(mws1gt_6,mws1gt_5),226).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),227). :- not holdsAt(moving(mws1gt_6,mws1gt_5),227).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),228). :- not holdsAt(moving(mws1gt_6,mws1gt_5),228).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),229). :- not holdsAt(moving(mws1gt_6,mws1gt_5),229).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),230). :- not holdsAt(moving(mws1gt_6,mws1gt_5),230).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),231). :- not holdsAt(moving(mws1gt_6,mws1gt_5),231).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),232). :- not holdsAt(moving(mws1gt_6,mws1gt_5),232).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),233). :- not holdsAt(moving(mws1gt_6,mws1gt_5),233).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),234). :- not holdsAt(moving(mws1gt_6,mws1gt_5),234).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),235). :- not holdsAt(moving(mws1gt_6,mws1gt_5),235).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),236). :- not holdsAt(moving(mws1gt_6,mws1gt_5),236).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),237). :- not holdsAt(moving(mws1gt_6,mws1gt_5),237).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),238). :- not holdsAt(moving(mws1gt_6,mws1gt_5),238).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),239). :- not holdsAt(moving(mws1gt_6,mws1gt_5),239).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),240). :- not holdsAt(moving(mws1gt_6,mws1gt_5),240).
:- holdsAt(moving(mws1gt_5,mws1gt_6),241). :- holdsAt(moving(mws1gt_6,mws1gt_5),241).
:- simpleEvent(T,P,inactive), T>216 , T<240.
:- simpleEvent(T,P,running), T>216 , T<240.

#show simpleEvent/3.