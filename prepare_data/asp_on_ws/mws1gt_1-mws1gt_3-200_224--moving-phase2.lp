

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
orientation(mws1gt_1,155,200). coords(mws1gt_1,140,133,200).  visible(mws1gt_1,200). orientation(mws1gt_3,162,200). coords(mws1gt_3,140,154,200).  visible(mws1gt_3,200).
orientation(mws1gt_1,155,201). coords(mws1gt_1,139,133,201).  visible(mws1gt_1,201). orientation(mws1gt_3,162,201). coords(mws1gt_3,140,154,201).  visible(mws1gt_3,201).
orientation(mws1gt_1,155,202). coords(mws1gt_1,139,133,202).  visible(mws1gt_1,202). orientation(mws1gt_3,162,202). coords(mws1gt_3,139,155,202).  visible(mws1gt_3,202).
orientation(mws1gt_1,155,203). coords(mws1gt_1,138,133,203).  visible(mws1gt_1,203). orientation(mws1gt_3,162,203). coords(mws1gt_3,139,155,203).  visible(mws1gt_3,203).
orientation(mws1gt_1,155,204). coords(mws1gt_1,138,133,204).  visible(mws1gt_1,204). orientation(mws1gt_3,162,204). coords(mws1gt_3,139,155,204).  visible(mws1gt_3,204).
orientation(mws1gt_1,155,205). coords(mws1gt_1,138,133,205).  visible(mws1gt_1,205). orientation(mws1gt_3,162,205). coords(mws1gt_3,139,155,205).  visible(mws1gt_3,205).
orientation(mws1gt_1,155,206). coords(mws1gt_1,137,132,206).  visible(mws1gt_1,206). orientation(mws1gt_3,162,206). coords(mws1gt_3,138,155,206).  visible(mws1gt_3,206).
orientation(mws1gt_1,155,207). coords(mws1gt_1,137,132,207).  visible(mws1gt_1,207). orientation(mws1gt_3,162,207). coords(mws1gt_3,138,155,207).  visible(mws1gt_3,207).
orientation(mws1gt_1,155,208). coords(mws1gt_1,137,132,208).  visible(mws1gt_1,208). orientation(mws1gt_3,162,208). coords(mws1gt_3,137,155,208).  visible(mws1gt_3,208).
orientation(mws1gt_1,155,209). coords(mws1gt_1,136,132,209).  visible(mws1gt_1,209). orientation(mws1gt_3,162,209). coords(mws1gt_3,135,155,209).  visible(mws1gt_3,209).
orientation(mws1gt_1,155,210). coords(mws1gt_1,135,132,210).  visible(mws1gt_1,210). orientation(mws1gt_3,162,210). coords(mws1gt_3,134,155,210).  visible(mws1gt_3,210).
orientation(mws1gt_1,155,211). coords(mws1gt_1,134,132,211).  visible(mws1gt_1,211). orientation(mws1gt_3,162,211). coords(mws1gt_3,132,155,211).  visible(mws1gt_3,211).
orientation(mws1gt_1,155,212). coords(mws1gt_1,134,132,212).  visible(mws1gt_1,212). orientation(mws1gt_3,162,212). coords(mws1gt_3,130,155,212).  visible(mws1gt_3,212).
orientation(mws1gt_1,155,213). coords(mws1gt_1,132,132,213).  visible(mws1gt_1,213). orientation(mws1gt_3,162,213). coords(mws1gt_3,127,155,213).  visible(mws1gt_3,213).
orientation(mws1gt_1,155,214). coords(mws1gt_1,131,131,214).  visible(mws1gt_1,214). orientation(mws1gt_3,162,214). coords(mws1gt_3,126,154,214).  visible(mws1gt_3,214).
orientation(mws1gt_1,155,215). coords(mws1gt_1,130,131,215).  visible(mws1gt_1,215). orientation(mws1gt_3,162,215). coords(mws1gt_3,125,154,215).  visible(mws1gt_3,215).
orientation(mws1gt_1,155,216). coords(mws1gt_1,128,132,216).  visible(mws1gt_1,216). orientation(mws1gt_3,162,216). coords(mws1gt_3,125,154,216).  visible(mws1gt_3,216).
orientation(mws1gt_1,155,217). coords(mws1gt_1,127,132,217).  visible(mws1gt_1,217). orientation(mws1gt_3,162,217). coords(mws1gt_3,125,154,217).  visible(mws1gt_3,217).
orientation(mws1gt_1,155,218). coords(mws1gt_1,126,132,218).  visible(mws1gt_1,218). orientation(mws1gt_3,162,218). coords(mws1gt_3,124,153,218).  visible(mws1gt_3,218).
orientation(mws1gt_1,155,219). coords(mws1gt_1,126,132,219).  visible(mws1gt_1,219). orientation(mws1gt_3,162,219). coords(mws1gt_3,124,153,219).  visible(mws1gt_3,219).
orientation(mws1gt_1,155,220). coords(mws1gt_1,126,132,220).  visible(mws1gt_1,220). orientation(mws1gt_3,162,220). coords(mws1gt_3,123,153,220).  visible(mws1gt_3,220).
orientation(mws1gt_1,155,221). coords(mws1gt_1,126,131,221).  visible(mws1gt_1,221). orientation(mws1gt_3,162,221). coords(mws1gt_3,123,153,221).  visible(mws1gt_3,221).
orientation(mws1gt_1,155,222). coords(mws1gt_1,125,131,222).  visible(mws1gt_1,222). orientation(mws1gt_3,162,222). coords(mws1gt_3,123,153,222).  visible(mws1gt_3,222).
orientation(mws1gt_1,155,223). coords(mws1gt_1,125,130,223).  visible(mws1gt_1,223). orientation(mws1gt_3,162,223). coords(mws1gt_3,122,152,223).  visible(mws1gt_3,223).
orientation(mws1gt_1,155,224). coords(mws1gt_1,125,130,224).  visible(mws1gt_1,224). orientation(mws1gt_3,162,224). coords(mws1gt_3,121,152,224).  visible(mws1gt_3,224).
orientation(mws1gt_1,155,225). coords(mws1gt_1,125,130,225).  visible(mws1gt_1,225). orientation(mws1gt_3,162,225). coords(mws1gt_3,121,152,225).  visible(mws1gt_3,225).
happensAt(disappear(mws1gt_1),224). happensAt(disappear(mws1gt_3),224).
1{simpleEvent(T,mws1gt_1,active); simpleEvent(T,mws1gt_1,inactive); simpleEvent(T,mws1gt_1,walking); simpleEvent(T,mws1gt_1,running)}1 :- T=200..224.
1{simpleEvent(T,mws1gt_3,active); simpleEvent(T,mws1gt_3,inactive); simpleEvent(T,mws1gt_3,walking); simpleEvent(T,mws1gt_3,running)}1 :- T=200..224.
:- holdsAt(moving(mws1gt_1,mws1gt_3),200). :- holdsAt(moving(mws1gt_3,mws1gt_1),200).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),201). :- not holdsAt(moving(mws1gt_3,mws1gt_1),201).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),202). :- not holdsAt(moving(mws1gt_3,mws1gt_1),202).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),203). :- not holdsAt(moving(mws1gt_3,mws1gt_1),203).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),204). :- not holdsAt(moving(mws1gt_3,mws1gt_1),204).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),205). :- not holdsAt(moving(mws1gt_3,mws1gt_1),205).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),206). :- not holdsAt(moving(mws1gt_3,mws1gt_1),206).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),207). :- not holdsAt(moving(mws1gt_3,mws1gt_1),207).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),208). :- not holdsAt(moving(mws1gt_3,mws1gt_1),208).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),209). :- not holdsAt(moving(mws1gt_3,mws1gt_1),209).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),210). :- not holdsAt(moving(mws1gt_3,mws1gt_1),210).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),211). :- not holdsAt(moving(mws1gt_3,mws1gt_1),211).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),212). :- not holdsAt(moving(mws1gt_3,mws1gt_1),212).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),213). :- not holdsAt(moving(mws1gt_3,mws1gt_1),213).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),214). :- not holdsAt(moving(mws1gt_3,mws1gt_1),214).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),215). :- not holdsAt(moving(mws1gt_3,mws1gt_1),215).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),216). :- not holdsAt(moving(mws1gt_3,mws1gt_1),216).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),217). :- not holdsAt(moving(mws1gt_3,mws1gt_1),217).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),218). :- not holdsAt(moving(mws1gt_3,mws1gt_1),218).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),219). :- not holdsAt(moving(mws1gt_3,mws1gt_1),219).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),220). :- not holdsAt(moving(mws1gt_3,mws1gt_1),220).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),221). :- not holdsAt(moving(mws1gt_3,mws1gt_1),221).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),222). :- not holdsAt(moving(mws1gt_3,mws1gt_1),222).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),223). :- not holdsAt(moving(mws1gt_3,mws1gt_1),223).
:- not holdsAt(moving(mws1gt_1,mws1gt_3),224). :- not holdsAt(moving(mws1gt_3,mws1gt_1),224).
:- holdsAt(moving(mws1gt_1,mws1gt_3),225). :- holdsAt(moving(mws1gt_3,mws1gt_1),225).
:- simpleEvent(T,P,inactive), T>200 , T<224.
:- simpleEvent(T,P,running), T>200 , T<224.

#show simpleEvent/3.