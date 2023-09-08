

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
orientation(mws1gt_5,161,204). coords(mws1gt_5,45,105,204).  visible(mws1gt_5,204). orientation(mws1gt_6,140,204). coords(mws1gt_6,60,88,204).  visible(mws1gt_6,204).
orientation(mws1gt_5,161,205). coords(mws1gt_5,45,107,205).  visible(mws1gt_5,205). orientation(mws1gt_6,146,205). coords(mws1gt_6,60,90,205).  visible(mws1gt_6,205).
orientation(mws1gt_5,161,206). coords(mws1gt_5,45,107,206).  visible(mws1gt_5,206). orientation(mws1gt_6,146,206). coords(mws1gt_6,59,91,206).  visible(mws1gt_6,206).
orientation(mws1gt_5,161,207). coords(mws1gt_5,45,107,207).  visible(mws1gt_5,207). orientation(mws1gt_6,146,207). coords(mws1gt_6,59,92,207).  visible(mws1gt_6,207).
orientation(mws1gt_5,161,208). coords(mws1gt_5,45,108,208).  visible(mws1gt_5,208). orientation(mws1gt_6,146,208). coords(mws1gt_6,59,92,208).  visible(mws1gt_6,208).
orientation(mws1gt_5,161,209). coords(mws1gt_5,45,109,209).  visible(mws1gt_5,209). orientation(mws1gt_6,146,209). coords(mws1gt_6,59,94,209).  visible(mws1gt_6,209).
orientation(mws1gt_5,161,210). coords(mws1gt_5,45,109,210).  visible(mws1gt_5,210). orientation(mws1gt_6,146,210). coords(mws1gt_6,59,94,210).  visible(mws1gt_6,210).
orientation(mws1gt_5,161,211). coords(mws1gt_5,45,109,211).  visible(mws1gt_5,211). orientation(mws1gt_6,146,211). coords(mws1gt_6,59,94,211).  visible(mws1gt_6,211).
orientation(mws1gt_5,161,212). coords(mws1gt_5,45,110,212).  visible(mws1gt_5,212). orientation(mws1gt_6,146,212). coords(mws1gt_6,59,95,212).  visible(mws1gt_6,212).
orientation(mws1gt_5,161,213). coords(mws1gt_5,45,111,213).  visible(mws1gt_5,213). orientation(mws1gt_6,146,213). coords(mws1gt_6,58,95,213).  visible(mws1gt_6,213).
orientation(mws1gt_5,161,214). coords(mws1gt_5,45,113,214).  visible(mws1gt_5,214). orientation(mws1gt_6,146,214). coords(mws1gt_6,57,95,214).  visible(mws1gt_6,214).
orientation(mws1gt_5,161,215). coords(mws1gt_5,46,116,215).  visible(mws1gt_5,215). orientation(mws1gt_6,146,215). coords(mws1gt_6,56,96,215).  visible(mws1gt_6,215).
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
orientation(mws1gt_5,161,229). coords(mws1gt_5,45,124,229).  visible(mws1gt_5,229). orientation(mws1gt_6,146,229). coords(mws1gt_6,52,101,229).  visible(mws1gt_6,229).
happensAt(disappear(mws1gt_5),228). happensAt(disappear(mws1gt_6),228).
1{simpleEvent(T,mws1gt_5,active); simpleEvent(T,mws1gt_5,inactive); simpleEvent(T,mws1gt_5,walking); simpleEvent(T,mws1gt_5,running)}1 :- T=204..228.
1{simpleEvent(T,mws1gt_6,active); simpleEvent(T,mws1gt_6,inactive); simpleEvent(T,mws1gt_6,walking); simpleEvent(T,mws1gt_6,running)}1 :- T=204..228.
:- holdsAt(moving(mws1gt_5,mws1gt_6),204). :- holdsAt(moving(mws1gt_6,mws1gt_5),204).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),205). :- not holdsAt(moving(mws1gt_6,mws1gt_5),205).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),206). :- not holdsAt(moving(mws1gt_6,mws1gt_5),206).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),207). :- not holdsAt(moving(mws1gt_6,mws1gt_5),207).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),208). :- not holdsAt(moving(mws1gt_6,mws1gt_5),208).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),209). :- not holdsAt(moving(mws1gt_6,mws1gt_5),209).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),210). :- not holdsAt(moving(mws1gt_6,mws1gt_5),210).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),211). :- not holdsAt(moving(mws1gt_6,mws1gt_5),211).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),212). :- not holdsAt(moving(mws1gt_6,mws1gt_5),212).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),213). :- not holdsAt(moving(mws1gt_6,mws1gt_5),213).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),214). :- not holdsAt(moving(mws1gt_6,mws1gt_5),214).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),215). :- not holdsAt(moving(mws1gt_6,mws1gt_5),215).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),216). :- not holdsAt(moving(mws1gt_6,mws1gt_5),216).
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
:- holdsAt(moving(mws1gt_5,mws1gt_6),229). :- holdsAt(moving(mws1gt_6,mws1gt_5),229).
:- simpleEvent(T,P,inactive), T>204 , T<228.
:- simpleEvent(T,P,running), T>204 , T<228.

#show simpleEvent/3.