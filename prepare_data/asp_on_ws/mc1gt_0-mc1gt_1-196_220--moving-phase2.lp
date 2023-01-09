

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
orientation(mc1gt_0,147,196). coords(mc1gt_0,131,117,196).  visible(mc1gt_0,196). orientation(mc1gt_1,140,196). coords(mc1gt_1,153,98,196).  visible(mc1gt_1,196).
orientation(mc1gt_0,147,197). coords(mc1gt_0,131,117,197).  visible(mc1gt_0,197). orientation(mc1gt_1,140,197). coords(mc1gt_1,153,100,197).  visible(mc1gt_1,197).
orientation(mc1gt_0,147,198). coords(mc1gt_0,131,117,198).  visible(mc1gt_0,198). orientation(mc1gt_1,139,198). coords(mc1gt_1,154,101,198).  visible(mc1gt_1,198).
orientation(mc1gt_0,147,199). coords(mc1gt_0,132,117,199).  visible(mc1gt_0,199). orientation(mc1gt_1,139,199). coords(mc1gt_1,155,103,199).  visible(mc1gt_1,199).
orientation(mc1gt_0,147,200). coords(mc1gt_0,134,117,200).  visible(mc1gt_0,200). orientation(mc1gt_1,139,200). coords(mc1gt_1,157,105,200).  visible(mc1gt_1,200).
orientation(mc1gt_0,147,201). coords(mc1gt_0,136,118,201).  visible(mc1gt_0,201). orientation(mc1gt_1,137,201). coords(mc1gt_1,159,106,201).  visible(mc1gt_1,201).
orientation(mc1gt_0,147,202). coords(mc1gt_0,138,118,202).  visible(mc1gt_0,202). orientation(mc1gt_1,137,202). coords(mc1gt_1,160,107,202).  visible(mc1gt_1,202).
orientation(mc1gt_0,147,203). coords(mc1gt_0,139,118,203).  visible(mc1gt_0,203). orientation(mc1gt_1,137,203). coords(mc1gt_1,161,108,203).  visible(mc1gt_1,203).
orientation(mc1gt_0,147,204). coords(mc1gt_0,140,118,204).  visible(mc1gt_0,204). orientation(mc1gt_1,137,204). coords(mc1gt_1,161,109,204).  visible(mc1gt_1,204).
orientation(mc1gt_0,147,205). coords(mc1gt_0,141,119,205).  visible(mc1gt_0,205). orientation(mc1gt_1,137,205). coords(mc1gt_1,162,109,205).  visible(mc1gt_1,205).
orientation(mc1gt_0,147,206). coords(mc1gt_0,141,119,206).  visible(mc1gt_0,206). orientation(mc1gt_1,140,206). coords(mc1gt_1,162,110,206).  visible(mc1gt_1,206).
orientation(mc1gt_0,147,207). coords(mc1gt_0,142,119,207).  visible(mc1gt_0,207). orientation(mc1gt_1,140,207). coords(mc1gt_1,162,110,207).  visible(mc1gt_1,207).
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
orientation(mc1gt_0,147,221). coords(mc1gt_0,151,128,221).  visible(mc1gt_0,221). orientation(mc1gt_1,141,221). coords(mc1gt_1,177,119,221).  visible(mc1gt_1,221).
happensAt(disappear(mc1gt_0),220). happensAt(disappear(mc1gt_1),220).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=196..220.
1{simpleEvent(T,mc1gt_1,active); simpleEvent(T,mc1gt_1,inactive); simpleEvent(T,mc1gt_1,walking); simpleEvent(T,mc1gt_1,running)}1 :- T=196..220.
:- holdsAt(moving(mc1gt_0,mc1gt_1),196). :- holdsAt(moving(mc1gt_1,mc1gt_0),196).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),197). :- not holdsAt(moving(mc1gt_1,mc1gt_0),197).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),198). :- not holdsAt(moving(mc1gt_1,mc1gt_0),198).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),199). :- not holdsAt(moving(mc1gt_1,mc1gt_0),199).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),200). :- not holdsAt(moving(mc1gt_1,mc1gt_0),200).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),201). :- not holdsAt(moving(mc1gt_1,mc1gt_0),201).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),202). :- not holdsAt(moving(mc1gt_1,mc1gt_0),202).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),203). :- not holdsAt(moving(mc1gt_1,mc1gt_0),203).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),204). :- not holdsAt(moving(mc1gt_1,mc1gt_0),204).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),205). :- not holdsAt(moving(mc1gt_1,mc1gt_0),205).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),206). :- not holdsAt(moving(mc1gt_1,mc1gt_0),206).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),207). :- not holdsAt(moving(mc1gt_1,mc1gt_0),207).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),208). :- not holdsAt(moving(mc1gt_1,mc1gt_0),208).
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
:- holdsAt(moving(mc1gt_0,mc1gt_1),221). :- holdsAt(moving(mc1gt_1,mc1gt_0),221).
:- simpleEvent(T,P,inactive), T>196 , T<220.
:- simpleEvent(T,P,running), T>196 , T<220.

#show simpleEvent/3.