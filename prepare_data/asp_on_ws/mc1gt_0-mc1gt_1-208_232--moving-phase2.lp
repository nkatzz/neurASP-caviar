

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

	person(mc1gt_0;mc1gt_1).
orientation(mc1gt_0,147,208). coords(mc1gt_0,142,119,208).  visible(mc1gt_0,208). orientation(mc1gt_1,140,208). coords(mc1gt_1,162,110,208).  visible(mc1gt_1,208).
orientation(mc1gt_0,147,209). coords(mc1gt_0,142,119,209).  visible(mc1gt_0,209). orientation(mc1gt_1,141,209). coords(mc1gt_1,163,111,209).  visible(mc1gt_1,209).
orientation(mc1gt_0,147,210). coords(mc1gt_0,143,119,210).  visible(mc1gt_0,210). orientation(mc1gt_1,141,210). coords(mc1gt_1,163,111,210).  visible(mc1gt_1,210).
orientation(mc1gt_0,147,211). coords(mc1gt_0,143,120,211).  visible(mc1gt_0,211). orientation(mc1gt_1,143,211). coords(mc1gt_1,163,111,211).  visible(mc1gt_1,211).
orientation(mc1gt_0,147,212). coords(mc1gt_0,143,121,212).  visible(mc1gt_0,212). orientation(mc1gt_1,142,212). coords(mc1gt_1,164,112,212).  visible(mc1gt_1,212).
orientation(mc1gt_0,147,213). coords(mc1gt_0,144,122,213).  visible(mc1gt_0,213). orientation(mc1gt_1,143,213). coords(mc1gt_1,167,112,213).  visible(mc1gt_1,213).
orientation(mc1gt_0,147,214). coords(mc1gt_0,144,123,214).  visible(mc1gt_0,214). orientation(mc1gt_1,143,214). coords(mc1gt_1,169,113,214).  visible(mc1gt_1,214).
orientation(mc1gt_0,147,215). coords(mc1gt_0,146,125,215).  visible(mc1gt_0,215). orientation(mc1gt_1,143,215). coords(mc1gt_1,171,114,215).  visible(mc1gt_1,215).
orientation(mc1gt_0,147,216). coords(mc1gt_0,147,126,216).  visible(mc1gt_0,216). orientation(mc1gt_1,142,216). coords(mc1gt_1,173,115,216).  visible(mc1gt_1,216).
orientation(mc1gt_0,147,217). coords(mc1gt_0,149,126,217).  visible(mc1gt_0,217). orientation(mc1gt_1,141,217). coords(mc1gt_1,174,116,217).  visible(mc1gt_1,217).
orientation(mc1gt_0,147,218). coords(mc1gt_0,150,127,218).  visible(mc1gt_0,218). orientation(mc1gt_1,141,218). coords(mc1gt_1,175,117,218).  visible(mc1gt_1,218).
orientation(mc1gt_0,147,219). coords(mc1gt_0,151,128,219).  visible(mc1gt_0,219). orientation(mc1gt_1,141,219). coords(mc1gt_1,176,118,219).  visible(mc1gt_1,219).
orientation(mc1gt_0,147,220). coords(mc1gt_0,151,128,220).  visible(mc1gt_0,220). orientation(mc1gt_1,141,220). coords(mc1gt_1,177,119,220).  visible(mc1gt_1,220).
orientation(mc1gt_0,147,221). coords(mc1gt_0,152,128,221).  visible(mc1gt_0,221). orientation(mc1gt_1,139,221). coords(mc1gt_1,178,119,221).  visible(mc1gt_1,221).
orientation(mc1gt_0,147,222). coords(mc1gt_0,152,129,222).  visible(mc1gt_0,222). orientation(mc1gt_1,138,222). coords(mc1gt_1,178,120,222).  visible(mc1gt_1,222).
orientation(mc1gt_0,147,223). coords(mc1gt_0,153,129,223).  visible(mc1gt_0,223). orientation(mc1gt_1,138,223). coords(mc1gt_1,178,120,223).  visible(mc1gt_1,223).
orientation(mc1gt_0,147,224). coords(mc1gt_0,153,130,224).  visible(mc1gt_0,224). orientation(mc1gt_1,138,224). coords(mc1gt_1,179,120,224).  visible(mc1gt_1,224).
orientation(mc1gt_0,147,225). coords(mc1gt_0,153,130,225).  visible(mc1gt_0,225). orientation(mc1gt_1,139,225). coords(mc1gt_1,179,121,225).  visible(mc1gt_1,225).
orientation(mc1gt_0,147,226). coords(mc1gt_0,154,130,226).  visible(mc1gt_0,226). orientation(mc1gt_1,142,226). coords(mc1gt_1,179,121,226).  visible(mc1gt_1,226).
orientation(mc1gt_0,147,227). coords(mc1gt_0,155,131,227).  visible(mc1gt_0,227). orientation(mc1gt_1,142,227). coords(mc1gt_1,179,123,227).  visible(mc1gt_1,227).
orientation(mc1gt_0,147,228). coords(mc1gt_0,157,131,228).  visible(mc1gt_0,228). orientation(mc1gt_1,142,228). coords(mc1gt_1,179,126,228).  visible(mc1gt_1,228).
orientation(mc1gt_0,147,229). coords(mc1gt_0,159,132,229).  visible(mc1gt_0,229). orientation(mc1gt_1,142,229). coords(mc1gt_1,181,129,229).  visible(mc1gt_1,229).
orientation(mc1gt_0,147,230). coords(mc1gt_0,160,133,230).  visible(mc1gt_0,230). orientation(mc1gt_1,142,230). coords(mc1gt_1,183,131,230).  visible(mc1gt_1,230).
orientation(mc1gt_0,147,231). coords(mc1gt_0,161,133,231).  visible(mc1gt_0,231). orientation(mc1gt_1,142,231). coords(mc1gt_1,184,133,231).  visible(mc1gt_1,231).
orientation(mc1gt_0,147,232). coords(mc1gt_0,162,135,232).  visible(mc1gt_0,232). orientation(mc1gt_1,142,232). coords(mc1gt_1,185,135,232).  visible(mc1gt_1,232).
orientation(mc1gt_0,147,233). coords(mc1gt_0,162,135,233).  visible(mc1gt_0,233). orientation(mc1gt_1,142,233). coords(mc1gt_1,185,135,233).  visible(mc1gt_1,233).
happensAt(disappear(mc1gt_0),232). happensAt(disappear(mc1gt_1),232).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=208..232.
1{simpleEvent(T,mc1gt_1,active); simpleEvent(T,mc1gt_1,inactive); simpleEvent(T,mc1gt_1,walking); simpleEvent(T,mc1gt_1,running)}1 :- T=208..232.
:- holdsAt(moving(mc1gt_0,mc1gt_1),208). :- holdsAt(moving(mc1gt_1,mc1gt_0),208).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),209). :- not holdsAt(moving(mc1gt_1,mc1gt_0),209).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),210). :- not holdsAt(moving(mc1gt_1,mc1gt_0),210).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),211). :- not holdsAt(moving(mc1gt_1,mc1gt_0),211).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),212). :- not holdsAt(moving(mc1gt_1,mc1gt_0),212).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),213). :- not holdsAt(moving(mc1gt_1,mc1gt_0),213).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),214). :- not holdsAt(moving(mc1gt_1,mc1gt_0),214).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),215). :- not holdsAt(moving(mc1gt_1,mc1gt_0),215).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),216). :- not holdsAt(moving(mc1gt_1,mc1gt_0),216).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),217). :- not holdsAt(moving(mc1gt_1,mc1gt_0),217).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),218). :- not holdsAt(moving(mc1gt_1,mc1gt_0),218).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),219). :- not holdsAt(moving(mc1gt_1,mc1gt_0),219).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),220). :- not holdsAt(moving(mc1gt_1,mc1gt_0),220).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),221). :- not holdsAt(moving(mc1gt_1,mc1gt_0),221).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),222). :- not holdsAt(moving(mc1gt_1,mc1gt_0),222).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),223). :- not holdsAt(moving(mc1gt_1,mc1gt_0),223).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),224). :- not holdsAt(moving(mc1gt_1,mc1gt_0),224).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),225). :- not holdsAt(moving(mc1gt_1,mc1gt_0),225).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),226). :- not holdsAt(moving(mc1gt_1,mc1gt_0),226).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),227). :- not holdsAt(moving(mc1gt_1,mc1gt_0),227).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),228). :- not holdsAt(moving(mc1gt_1,mc1gt_0),228).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),229). :- not holdsAt(moving(mc1gt_1,mc1gt_0),229).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),230). :- not holdsAt(moving(mc1gt_1,mc1gt_0),230).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),231). :- not holdsAt(moving(mc1gt_1,mc1gt_0),231).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),232). :- not holdsAt(moving(mc1gt_1,mc1gt_0),232).
:- holdsAt(moving(mc1gt_0,mc1gt_1),233). :- holdsAt(moving(mc1gt_1,mc1gt_0),233).
:- simpleEvent(T,P,inactive), T>208 , T<232.
:- simpleEvent(T,P,running), T>208 , T<232.

#show simpleEvent/3.