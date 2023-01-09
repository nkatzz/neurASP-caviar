

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

	person(spgt_0;spgt_1).
orientation(spgt_0,152,108). coords(spgt_0,77,147,108).  visible(spgt_0,108). orientation(spgt_1,154,108). coords(spgt_1,78,134,108).  visible(spgt_1,108).
orientation(spgt_0,152,109). coords(spgt_0,77,146,109).  visible(spgt_0,109). orientation(spgt_1,154,109). coords(spgt_1,78,134,109).  visible(spgt_1,109).
orientation(spgt_0,152,110). coords(spgt_0,78,145,110).  visible(spgt_0,110). orientation(spgt_1,154,110). coords(spgt_1,78,132,110).  visible(spgt_1,110).
orientation(spgt_0,152,111). coords(spgt_0,78,145,111).  visible(spgt_0,111). orientation(spgt_1,154,111). coords(spgt_1,78,132,111).  visible(spgt_1,111).
orientation(spgt_0,152,112). coords(spgt_0,78,144,112).  visible(spgt_0,112). orientation(spgt_1,154,112). coords(spgt_1,79,131,112).  visible(spgt_1,112).
orientation(spgt_0,152,113). coords(spgt_0,79,144,113).  visible(spgt_0,113). orientation(spgt_1,154,113). coords(spgt_1,79,130,113).  visible(spgt_1,113).
orientation(spgt_0,152,114). coords(spgt_0,79,143,114).  visible(spgt_0,114). orientation(spgt_1,154,114). coords(spgt_1,79,129,114).  visible(spgt_1,114).
orientation(spgt_0,152,115). coords(spgt_0,82,142,115).  visible(spgt_0,115). orientation(spgt_1,154,115). coords(spgt_1,81,128,115).  visible(spgt_1,115).
orientation(spgt_0,152,116). coords(spgt_0,82,142,116).  visible(spgt_0,116). orientation(spgt_1,154,116). coords(spgt_1,82,128,116).  visible(spgt_1,116).
orientation(spgt_0,152,117). coords(spgt_0,83,142,117).  visible(spgt_0,117). orientation(spgt_1,154,117). coords(spgt_1,82,128,117).  visible(spgt_1,117).
orientation(spgt_0,152,118). coords(spgt_0,84,141,118).  visible(spgt_0,118). orientation(spgt_1,154,118). coords(spgt_1,84,127,118).  visible(spgt_1,118).
orientation(spgt_0,152,119). coords(spgt_0,85,141,119).  visible(spgt_0,119). orientation(spgt_1,154,119). coords(spgt_1,84,127,119).  visible(spgt_1,119).
orientation(spgt_0,152,120). coords(spgt_0,85,140,120).  visible(spgt_0,120). orientation(spgt_1,154,120). coords(spgt_1,84,127,120).  visible(spgt_1,120).
orientation(spgt_0,152,121). coords(spgt_0,86,140,121).  visible(spgt_0,121). orientation(spgt_1,154,121). coords(spgt_1,85,127,121).  visible(spgt_1,121).
orientation(spgt_0,152,122). coords(spgt_0,86,140,122).  visible(spgt_0,122). orientation(spgt_1,154,122). coords(spgt_1,86,126,122).  visible(spgt_1,122).
orientation(spgt_0,152,123). coords(spgt_0,86,139,123).  visible(spgt_0,123). orientation(spgt_1,154,123). coords(spgt_1,86,125,123).  visible(spgt_1,123).
orientation(spgt_0,152,124). coords(spgt_0,87,139,124).  visible(spgt_0,124). orientation(spgt_1,154,124). coords(spgt_1,87,124,124).  visible(spgt_1,124).
orientation(spgt_0,152,125). coords(spgt_0,87,138,125).  visible(spgt_0,125). orientation(spgt_1,154,125). coords(spgt_1,87,124,125).  visible(spgt_1,125).
orientation(spgt_0,152,126). coords(spgt_0,88,138,126).  visible(spgt_0,126). orientation(spgt_1,154,126). coords(spgt_1,87,124,126).  visible(spgt_1,126).
orientation(spgt_0,152,127). coords(spgt_0,89,138,127).  visible(spgt_0,127). orientation(spgt_1,154,127). coords(spgt_1,87,123,127).  visible(spgt_1,127).
orientation(spgt_0,152,128). coords(spgt_0,91,137,128).  visible(spgt_0,128). orientation(spgt_1,154,128). coords(spgt_1,89,123,128).  visible(spgt_1,128).
orientation(spgt_0,152,129). coords(spgt_0,92,137,129).  visible(spgt_0,129). orientation(spgt_1,154,129). coords(spgt_1,91,122,129).  visible(spgt_1,129).
orientation(spgt_0,152,130). coords(spgt_0,92,137,130).  visible(spgt_0,130). orientation(spgt_1,154,130). coords(spgt_1,91,122,130).  visible(spgt_1,130).
orientation(spgt_0,152,131). coords(spgt_0,93,137,131).  visible(spgt_0,131). orientation(spgt_1,154,131). coords(spgt_1,92,122,131).  visible(spgt_1,131).
orientation(spgt_0,152,132). coords(spgt_0,95,136,132).  visible(spgt_0,132). orientation(spgt_1,154,132). coords(spgt_1,93,121,132).  visible(spgt_1,132).
orientation(spgt_0,152,133). coords(spgt_0,95,136,133).  visible(spgt_0,133). orientation(spgt_1,154,133). coords(spgt_1,93,121,133).  visible(spgt_1,133).
happensAt(disappear(spgt_0),132). happensAt(disappear(spgt_1),132).
1{simpleEvent(T,spgt_0,active); simpleEvent(T,spgt_0,inactive); simpleEvent(T,spgt_0,walking); simpleEvent(T,spgt_0,running)}1 :- T=108..132.
1{simpleEvent(T,spgt_1,active); simpleEvent(T,spgt_1,inactive); simpleEvent(T,spgt_1,walking); simpleEvent(T,spgt_1,running)}1 :- T=108..132.
:- holdsAt(moving(spgt_0,spgt_1),108). :- holdsAt(moving(spgt_1,spgt_0),108).
:- not holdsAt(moving(spgt_0,spgt_1),109). :- not holdsAt(moving(spgt_1,spgt_0),109).
:- not holdsAt(moving(spgt_0,spgt_1),110). :- not holdsAt(moving(spgt_1,spgt_0),110).
:- not holdsAt(moving(spgt_0,spgt_1),111). :- not holdsAt(moving(spgt_1,spgt_0),111).
:- not holdsAt(moving(spgt_0,spgt_1),112). :- not holdsAt(moving(spgt_1,spgt_0),112).
:- not holdsAt(moving(spgt_0,spgt_1),113). :- not holdsAt(moving(spgt_1,spgt_0),113).
:- not holdsAt(moving(spgt_0,spgt_1),114). :- not holdsAt(moving(spgt_1,spgt_0),114).
:- not holdsAt(moving(spgt_0,spgt_1),115). :- not holdsAt(moving(spgt_1,spgt_0),115).
:- not holdsAt(moving(spgt_0,spgt_1),116). :- not holdsAt(moving(spgt_1,spgt_0),116).
:- not holdsAt(moving(spgt_0,spgt_1),117). :- not holdsAt(moving(spgt_1,spgt_0),117).
:- not holdsAt(moving(spgt_0,spgt_1),118). :- not holdsAt(moving(spgt_1,spgt_0),118).
:- not holdsAt(moving(spgt_0,spgt_1),119). :- not holdsAt(moving(spgt_1,spgt_0),119).
:- not holdsAt(moving(spgt_0,spgt_1),120). :- not holdsAt(moving(spgt_1,spgt_0),120).
:- not holdsAt(moving(spgt_0,spgt_1),121). :- not holdsAt(moving(spgt_1,spgt_0),121).
:- not holdsAt(moving(spgt_0,spgt_1),122). :- not holdsAt(moving(spgt_1,spgt_0),122).
:- not holdsAt(moving(spgt_0,spgt_1),123). :- not holdsAt(moving(spgt_1,spgt_0),123).
:- not holdsAt(moving(spgt_0,spgt_1),124). :- not holdsAt(moving(spgt_1,spgt_0),124).
:- not holdsAt(moving(spgt_0,spgt_1),125). :- not holdsAt(moving(spgt_1,spgt_0),125).
:- not holdsAt(moving(spgt_0,spgt_1),126). :- not holdsAt(moving(spgt_1,spgt_0),126).
:- not holdsAt(moving(spgt_0,spgt_1),127). :- not holdsAt(moving(spgt_1,spgt_0),127).
:- not holdsAt(moving(spgt_0,spgt_1),128). :- not holdsAt(moving(spgt_1,spgt_0),128).
:- not holdsAt(moving(spgt_0,spgt_1),129). :- not holdsAt(moving(spgt_1,spgt_0),129).
:- not holdsAt(moving(spgt_0,spgt_1),130). :- not holdsAt(moving(spgt_1,spgt_0),130).
:- not holdsAt(moving(spgt_0,spgt_1),131). :- not holdsAt(moving(spgt_1,spgt_0),131).
:- not holdsAt(moving(spgt_0,spgt_1),132). :- not holdsAt(moving(spgt_1,spgt_0),132).
:- holdsAt(moving(spgt_0,spgt_1),133). :- holdsAt(moving(spgt_1,spgt_0),133).
:- simpleEvent(T,P,inactive), T>108 , T<132.
:- simpleEvent(T,P,running), T>108 , T<132.

#show simpleEvent/3.