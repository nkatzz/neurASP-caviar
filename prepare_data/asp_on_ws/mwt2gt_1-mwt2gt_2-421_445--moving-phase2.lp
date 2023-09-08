

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

	person(mwt2gt_1;mwt2gt_2).
orientation(mwt2gt_1,150,421). coords(mwt2gt_1,94,89,421).  visible(mwt2gt_1,421). orientation(mwt2gt_2,149,421). coords(mwt2gt_2,90,122,421).  visible(mwt2gt_2,421).
orientation(mwt2gt_1,150,422). coords(mwt2gt_1,94,89,422).  visible(mwt2gt_1,422). orientation(mwt2gt_2,149,422). coords(mwt2gt_2,88,122,422).  visible(mwt2gt_2,422).
orientation(mwt2gt_1,151,423). coords(mwt2gt_1,93,89,423).  visible(mwt2gt_1,423). orientation(mwt2gt_2,149,423). coords(mwt2gt_2,87,122,423).  visible(mwt2gt_2,423).
orientation(mwt2gt_1,153,424). coords(mwt2gt_1,92,89,424).  visible(mwt2gt_1,424). orientation(mwt2gt_2,149,424). coords(mwt2gt_2,87,122,424).  visible(mwt2gt_2,424).
orientation(mwt2gt_1,153,425). coords(mwt2gt_1,91,90,425).  visible(mwt2gt_1,425). orientation(mwt2gt_2,149,425). coords(mwt2gt_2,86,122,425).  visible(mwt2gt_2,425).
orientation(mwt2gt_1,153,426). coords(mwt2gt_1,90,90,426).  visible(mwt2gt_1,426). orientation(mwt2gt_2,149,426). coords(mwt2gt_2,85,122,426).  visible(mwt2gt_2,426).
orientation(mwt2gt_1,153,427). coords(mwt2gt_1,90,90,427).  visible(mwt2gt_1,427). orientation(mwt2gt_2,149,427). coords(mwt2gt_2,85,122,427).  visible(mwt2gt_2,427).
orientation(mwt2gt_1,153,428). coords(mwt2gt_1,89,90,428).  visible(mwt2gt_1,428). orientation(mwt2gt_2,150,428). coords(mwt2gt_2,85,122,428).  visible(mwt2gt_2,428).
orientation(mwt2gt_1,153,429). coords(mwt2gt_1,89,90,429).  visible(mwt2gt_1,429). orientation(mwt2gt_2,150,429). coords(mwt2gt_2,85,122,429).  visible(mwt2gt_2,429).
orientation(mwt2gt_1,153,430). coords(mwt2gt_1,89,91,430).  visible(mwt2gt_1,430). orientation(mwt2gt_2,150,430). coords(mwt2gt_2,85,122,430).  visible(mwt2gt_2,430).
orientation(mwt2gt_1,153,431). coords(mwt2gt_1,89,91,431).  visible(mwt2gt_1,431). orientation(mwt2gt_2,150,431). coords(mwt2gt_2,85,122,431).  visible(mwt2gt_2,431).
orientation(mwt2gt_1,153,432). coords(mwt2gt_1,88,92,432).  visible(mwt2gt_1,432). orientation(mwt2gt_2,150,432). coords(mwt2gt_2,85,122,432).  visible(mwt2gt_2,432).
orientation(mwt2gt_1,153,433). coords(mwt2gt_1,87,92,433).  visible(mwt2gt_1,433). orientation(mwt2gt_2,150,433). coords(mwt2gt_2,84,122,433).  visible(mwt2gt_2,433).
orientation(mwt2gt_1,153,434). coords(mwt2gt_1,86,92,434).  visible(mwt2gt_1,434). orientation(mwt2gt_2,150,434). coords(mwt2gt_2,83,123,434).  visible(mwt2gt_2,434).
orientation(mwt2gt_1,153,435). coords(mwt2gt_1,86,93,435).  visible(mwt2gt_1,435). orientation(mwt2gt_2,150,435). coords(mwt2gt_2,83,123,435).  visible(mwt2gt_2,435).
orientation(mwt2gt_1,153,436). coords(mwt2gt_1,85,94,436).  visible(mwt2gt_1,436). orientation(mwt2gt_2,150,436). coords(mwt2gt_2,82,123,436).  visible(mwt2gt_2,436).
orientation(mwt2gt_1,153,437). coords(mwt2gt_1,85,95,437).  visible(mwt2gt_1,437). orientation(mwt2gt_2,150,437). coords(mwt2gt_2,82,123,437).  visible(mwt2gt_2,437).
orientation(mwt2gt_1,153,438). coords(mwt2gt_1,85,97,438).  visible(mwt2gt_1,438). orientation(mwt2gt_2,150,438). coords(mwt2gt_2,81,123,438).  visible(mwt2gt_2,438).
orientation(mwt2gt_1,150,439). coords(mwt2gt_1,84,98,439).  visible(mwt2gt_1,439). orientation(mwt2gt_2,150,439). coords(mwt2gt_2,81,123,439).  visible(mwt2gt_2,439).
orientation(mwt2gt_1,150,440). coords(mwt2gt_1,84,99,440).  visible(mwt2gt_1,440). orientation(mwt2gt_2,150,440). coords(mwt2gt_2,81,124,440).  visible(mwt2gt_2,440).
orientation(mwt2gt_1,155,441). coords(mwt2gt_1,84,101,441).  visible(mwt2gt_1,441). orientation(mwt2gt_2,150,441). coords(mwt2gt_2,81,125,441).  visible(mwt2gt_2,441).
orientation(mwt2gt_1,155,442). coords(mwt2gt_1,83,101,442).  visible(mwt2gt_1,442). orientation(mwt2gt_2,150,442). coords(mwt2gt_2,80,126,442).  visible(mwt2gt_2,442).
orientation(mwt2gt_1,155,443). coords(mwt2gt_1,83,102,443).  visible(mwt2gt_1,443). orientation(mwt2gt_2,150,443). coords(mwt2gt_2,80,127,443).  visible(mwt2gt_2,443).
orientation(mwt2gt_1,155,444). coords(mwt2gt_1,82,102,444).  visible(mwt2gt_1,444). orientation(mwt2gt_2,150,444). coords(mwt2gt_2,80,128,444).  visible(mwt2gt_2,444).
orientation(mwt2gt_1,155,445). coords(mwt2gt_1,81,103,445).  visible(mwt2gt_1,445). orientation(mwt2gt_2,150,445). coords(mwt2gt_2,80,128,445).  visible(mwt2gt_2,445).
orientation(mwt2gt_1,155,446). coords(mwt2gt_1,81,103,446).  visible(mwt2gt_1,446). orientation(mwt2gt_2,150,446). coords(mwt2gt_2,80,128,446).  visible(mwt2gt_2,446).
happensAt(disappear(mwt2gt_1),445). happensAt(disappear(mwt2gt_2),445).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=421..445.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=421..445.
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),421). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),421).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),422). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),422).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),423). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),423).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),424). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),424).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),425). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),425).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),426). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),426).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),427). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),427).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),428). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),428).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),429). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),429).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),430). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),430).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),431). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),431).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),432). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),432).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),433). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),433).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),434). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),434).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),435). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),435).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),436). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),436).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),437). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),437).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),438). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),438).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),439). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),439).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),440). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),440).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),441). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),441).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),442). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),442).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),443). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),443).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),444). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),444).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),445). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),445).
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),446). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),446).
:- simpleEvent(T,P,inactive), T>421 , T<445.
:- simpleEvent(T,P,running), T>421 , T<445.

#show simpleEvent/3.