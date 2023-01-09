

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
orientation(mwt2gt_1,155,457). coords(mwt2gt_1,74,115,457).  visible(mwt2gt_1,457). orientation(mwt2gt_2,157,457). coords(mwt2gt_2,72,135,457).  visible(mwt2gt_2,457).
orientation(mwt2gt_1,155,458). coords(mwt2gt_1,73,115,458).  visible(mwt2gt_1,458). orientation(mwt2gt_2,157,458). coords(mwt2gt_2,71,136,458).  visible(mwt2gt_2,458).
orientation(mwt2gt_1,155,459). coords(mwt2gt_1,72,116,459).  visible(mwt2gt_1,459). orientation(mwt2gt_2,157,459). coords(mwt2gt_2,71,137,459).  visible(mwt2gt_2,459).
orientation(mwt2gt_1,155,460). coords(mwt2gt_1,71,116,460).  visible(mwt2gt_1,460). orientation(mwt2gt_2,157,460). coords(mwt2gt_2,71,137,460).  visible(mwt2gt_2,460).
orientation(mwt2gt_1,155,461). coords(mwt2gt_1,70,118,461).  visible(mwt2gt_1,461). orientation(mwt2gt_2,157,461). coords(mwt2gt_2,70,138,461).  visible(mwt2gt_2,461).
orientation(mwt2gt_1,155,462). coords(mwt2gt_1,70,120,462).  visible(mwt2gt_1,462). orientation(mwt2gt_2,157,462). coords(mwt2gt_2,69,138,462).  visible(mwt2gt_2,462).
orientation(mwt2gt_1,155,463). coords(mwt2gt_1,69,122,463).  visible(mwt2gt_1,463). orientation(mwt2gt_2,157,463). coords(mwt2gt_2,68,138,463).  visible(mwt2gt_2,463).
orientation(mwt2gt_1,153,464). coords(mwt2gt_1,70,123,464).  visible(mwt2gt_1,464). orientation(mwt2gt_2,157,464). coords(mwt2gt_2,68,139,464).  visible(mwt2gt_2,464).
orientation(mwt2gt_1,153,465). coords(mwt2gt_1,69,125,465).  visible(mwt2gt_1,465). orientation(mwt2gt_2,157,465). coords(mwt2gt_2,66,139,465).  visible(mwt2gt_2,465).
orientation(mwt2gt_1,160,466). coords(mwt2gt_1,69,126,466).  visible(mwt2gt_1,466). orientation(mwt2gt_2,157,466). coords(mwt2gt_2,65,141,466).  visible(mwt2gt_2,466).
orientation(mwt2gt_1,160,467). coords(mwt2gt_1,68,126,467).  visible(mwt2gt_1,467). orientation(mwt2gt_2,157,467). coords(mwt2gt_2,65,143,467).  visible(mwt2gt_2,467).
orientation(mwt2gt_1,160,468). coords(mwt2gt_1,68,127,468).  visible(mwt2gt_1,468). orientation(mwt2gt_2,157,468). coords(mwt2gt_2,65,144,468).  visible(mwt2gt_2,468).
orientation(mwt2gt_1,164,469). coords(mwt2gt_1,66,128,469).  visible(mwt2gt_1,469). orientation(mwt2gt_2,157,469). coords(mwt2gt_2,65,146,469).  visible(mwt2gt_2,469).
orientation(mwt2gt_1,164,470). coords(mwt2gt_1,66,128,470).  visible(mwt2gt_1,470). orientation(mwt2gt_2,157,470). coords(mwt2gt_2,65,146,470).  visible(mwt2gt_2,470).
orientation(mwt2gt_1,164,471). coords(mwt2gt_1,65,129,471).  visible(mwt2gt_1,471). orientation(mwt2gt_2,157,471). coords(mwt2gt_2,65,147,471).  visible(mwt2gt_2,471).
orientation(mwt2gt_1,164,472). coords(mwt2gt_1,65,129,472).  visible(mwt2gt_1,472). orientation(mwt2gt_2,144,472). coords(mwt2gt_2,64,148,472).  visible(mwt2gt_2,472).
orientation(mwt2gt_1,164,473). coords(mwt2gt_1,64,129,473).  visible(mwt2gt_1,473). orientation(mwt2gt_2,144,473). coords(mwt2gt_2,64,148,473).  visible(mwt2gt_2,473).
orientation(mwt2gt_1,164,474). coords(mwt2gt_1,64,130,474).  visible(mwt2gt_1,474). orientation(mwt2gt_2,152,474). coords(mwt2gt_2,63,149,474).  visible(mwt2gt_2,474).
orientation(mwt2gt_1,164,475). coords(mwt2gt_1,64,132,475).  visible(mwt2gt_1,475). orientation(mwt2gt_2,153,475). coords(mwt2gt_2,62,149,475).  visible(mwt2gt_2,475).
orientation(mwt2gt_1,164,476). coords(mwt2gt_1,64,134,476).  visible(mwt2gt_1,476). orientation(mwt2gt_2,153,476). coords(mwt2gt_2,61,150,476).  visible(mwt2gt_2,476).
orientation(mwt2gt_1,164,477). coords(mwt2gt_1,64,135,477).  visible(mwt2gt_1,477). orientation(mwt2gt_2,150,477). coords(mwt2gt_2,60,150,477).  visible(mwt2gt_2,477).
orientation(mwt2gt_1,164,478). coords(mwt2gt_1,63,137,478).  visible(mwt2gt_1,478). orientation(mwt2gt_2,157,478). coords(mwt2gt_2,59,150,478).  visible(mwt2gt_2,478).
orientation(mwt2gt_1,164,479). coords(mwt2gt_1,64,137,479).  visible(mwt2gt_1,479). orientation(mwt2gt_2,162,479). coords(mwt2gt_2,59,150,479).  visible(mwt2gt_2,479).
orientation(mwt2gt_1,164,480). coords(mwt2gt_1,63,138,480).  visible(mwt2gt_1,480). orientation(mwt2gt_2,161,480). coords(mwt2gt_2,59,150,480).  visible(mwt2gt_2,480).
orientation(mwt2gt_1,164,481). coords(mwt2gt_1,62,138,481).  visible(mwt2gt_1,481). orientation(mwt2gt_2,160,481). coords(mwt2gt_2,59,151,481).  visible(mwt2gt_2,481).
orientation(mwt2gt_1,164,482). coords(mwt2gt_1,62,138,482).  visible(mwt2gt_1,482). orientation(mwt2gt_2,160,482). coords(mwt2gt_2,59,151,482).  visible(mwt2gt_2,482).
happensAt(disappear(mwt2gt_1),481). happensAt(disappear(mwt2gt_2),481).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=457..481.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=457..481.
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),457). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),457).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),458). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),458).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),459). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),459).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),460). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),460).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),461). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),461).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),462). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),462).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),463). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),463).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),464). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),464).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),465). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),465).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),466). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),466).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),467). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),467).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),468). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),468).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),469). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),469).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),470). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),470).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),471). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),471).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),472). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),472).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),473). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),473).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),474). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),474).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),475). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),475).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),476). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),476).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),477). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),477).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),478). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),478).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),479). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),479).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),480). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),480).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),481). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),481).
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),482). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),482).
:- simpleEvent(T,P,inactive), T>457 , T<481.
:- simpleEvent(T,P,running), T>457 , T<481.

#show simpleEvent/3.