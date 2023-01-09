

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
orientation(wk2gt_1,125,840). coords(wk2gt_1,115,53,840).  visible(wk2gt_1,840). orientation(wk2gt_2,142,840). coords(wk2gt_2,99,63,840).  visible(wk2gt_2,840).
orientation(wk2gt_1,125,841). coords(wk2gt_1,115,53,841).  visible(wk2gt_1,841). orientation(wk2gt_2,142,841). coords(wk2gt_2,99,63,841).  visible(wk2gt_2,841).
orientation(wk2gt_1,125,842). coords(wk2gt_1,115,53,842).  visible(wk2gt_1,842). orientation(wk2gt_2,142,842). coords(wk2gt_2,99,63,842).  visible(wk2gt_2,842).
orientation(wk2gt_1,125,843). coords(wk2gt_1,115,53,843).  visible(wk2gt_1,843). orientation(wk2gt_2,142,843). coords(wk2gt_2,99,63,843).  visible(wk2gt_2,843).
orientation(wk2gt_1,125,844). coords(wk2gt_1,115,53,844).  visible(wk2gt_1,844). orientation(wk2gt_2,142,844). coords(wk2gt_2,99,63,844).  visible(wk2gt_2,844).
orientation(wk2gt_1,125,845). coords(wk2gt_1,115,53,845).  visible(wk2gt_1,845). orientation(wk2gt_2,142,845). coords(wk2gt_2,99,63,845).  visible(wk2gt_2,845).
orientation(wk2gt_1,125,846). coords(wk2gt_1,115,53,846).  visible(wk2gt_1,846). orientation(wk2gt_2,142,846). coords(wk2gt_2,99,63,846).  visible(wk2gt_2,846).
orientation(wk2gt_1,127,847). coords(wk2gt_1,115,53,847).  visible(wk2gt_1,847). orientation(wk2gt_2,142,847). coords(wk2gt_2,99,63,847).  visible(wk2gt_2,847).
orientation(wk2gt_1,127,848). coords(wk2gt_1,115,53,848).  visible(wk2gt_1,848). orientation(wk2gt_2,142,848). coords(wk2gt_2,99,63,848).  visible(wk2gt_2,848).
orientation(wk2gt_1,127,849). coords(wk2gt_1,115,53,849).  visible(wk2gt_1,849). orientation(wk2gt_2,142,849). coords(wk2gt_2,99,63,849).  visible(wk2gt_2,849).
orientation(wk2gt_1,127,850). coords(wk2gt_1,115,53,850).  visible(wk2gt_1,850). orientation(wk2gt_2,142,850). coords(wk2gt_2,99,63,850).  visible(wk2gt_2,850).
orientation(wk2gt_1,127,851). coords(wk2gt_1,115,53,851).  visible(wk2gt_1,851). orientation(wk2gt_2,142,851). coords(wk2gt_2,99,63,851).  visible(wk2gt_2,851).
orientation(wk2gt_1,127,852). coords(wk2gt_1,115,53,852).  visible(wk2gt_1,852). orientation(wk2gt_2,142,852). coords(wk2gt_2,99,63,852).  visible(wk2gt_2,852).
orientation(wk2gt_1,127,853). coords(wk2gt_1,115,53,853).  visible(wk2gt_1,853). orientation(wk2gt_2,142,853). coords(wk2gt_2,99,63,853).  visible(wk2gt_2,853).
orientation(wk2gt_1,127,854). coords(wk2gt_1,115,53,854).  visible(wk2gt_1,854). orientation(wk2gt_2,142,854). coords(wk2gt_2,99,63,854).  visible(wk2gt_2,854).
orientation(wk2gt_1,127,855). coords(wk2gt_1,115,53,855).  visible(wk2gt_1,855). orientation(wk2gt_2,142,855). coords(wk2gt_2,99,63,855).  visible(wk2gt_2,855).
orientation(wk2gt_1,127,856). coords(wk2gt_1,115,53,856).  visible(wk2gt_1,856). orientation(wk2gt_2,142,856). coords(wk2gt_2,99,63,856).  visible(wk2gt_2,856).
orientation(wk2gt_1,127,857). coords(wk2gt_1,115,53,857).  visible(wk2gt_1,857). orientation(wk2gt_2,142,857). coords(wk2gt_2,99,63,857).  visible(wk2gt_2,857).
orientation(wk2gt_1,127,858). coords(wk2gt_1,115,53,858).  visible(wk2gt_1,858). orientation(wk2gt_2,142,858). coords(wk2gt_2,99,63,858).  visible(wk2gt_2,858).
orientation(wk2gt_1,127,859). coords(wk2gt_1,115,53,859).  visible(wk2gt_1,859). orientation(wk2gt_2,142,859). coords(wk2gt_2,99,63,859).  visible(wk2gt_2,859).
orientation(wk2gt_1,127,860). coords(wk2gt_1,115,53,860).  visible(wk2gt_1,860). orientation(wk2gt_2,142,860). coords(wk2gt_2,99,63,860).  visible(wk2gt_2,860).
orientation(wk2gt_1,127,861). coords(wk2gt_1,115,53,861).  visible(wk2gt_1,861). orientation(wk2gt_2,142,861). coords(wk2gt_2,99,63,861).  visible(wk2gt_2,861).
orientation(wk2gt_1,127,862). coords(wk2gt_1,115,53,862).  visible(wk2gt_1,862). orientation(wk2gt_2,142,862). coords(wk2gt_2,99,63,862).  visible(wk2gt_2,862).
orientation(wk2gt_1,127,863). coords(wk2gt_1,115,53,863).  visible(wk2gt_1,863). orientation(wk2gt_2,142,863). coords(wk2gt_2,99,63,863).  visible(wk2gt_2,863).
orientation(wk2gt_1,127,864). coords(wk2gt_1,115,53,864).  visible(wk2gt_1,864). orientation(wk2gt_2,142,864). coords(wk2gt_2,99,63,864).  visible(wk2gt_2,864).
orientation(wk2gt_1,127,865). coords(wk2gt_1,115,53,865).  visible(wk2gt_1,865). orientation(wk2gt_2,142,865). coords(wk2gt_2,99,63,865).  visible(wk2gt_2,865).
happensAt(disappear(wk2gt_1),864). happensAt(disappear(wk2gt_2),864).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=840..864.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=840..864.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),840). :- holdsAt(meeting(wk2gt_2,wk2gt_1),840).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),841). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),841).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),842). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),842).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),843). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),843).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),844). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),844).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),845). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),845).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),846). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),846).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),847). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),847).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),848). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),848).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),849). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),849).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),850). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),850).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),851). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),851).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),852). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),852).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),853). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),853).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),854). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),854).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),855). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),855).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),856). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),856).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),857). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),857).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),858). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),858).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),859). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),859).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),860). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),860).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),861). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),861).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),862). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),862).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),863). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),863).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),864). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),864).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),865). :- holdsAt(meeting(wk2gt_2,wk2gt_1),865).
:- simpleEvent(T,P,walking), T>840 , T<864.
:- simpleEvent(T,P,running), T>840 , T<864.

#show simpleEvent/3.