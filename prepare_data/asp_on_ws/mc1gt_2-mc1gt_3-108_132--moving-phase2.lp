

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

	person(mc1gt_2;mc1gt_3).
orientation(mc1gt_2,144,108). coords(mc1gt_2,76,73,108).  visible(mc1gt_2,108). orientation(mc1gt_3,146,108). coords(mc1gt_3,83,65,108).  happensAt(appear(mc1gt_3),108).
orientation(mc1gt_2,144,109). coords(mc1gt_2,76,74,109).  visible(mc1gt_2,109). orientation(mc1gt_3,146,109). coords(mc1gt_3,79,65,109).  visible(mc1gt_3,109).
orientation(mc1gt_2,144,110). coords(mc1gt_2,76,74,110).  visible(mc1gt_2,110). orientation(mc1gt_3,146,110). coords(mc1gt_3,79,66,110).  visible(mc1gt_3,110).
orientation(mc1gt_2,144,111). coords(mc1gt_2,77,74,111).  visible(mc1gt_2,111). orientation(mc1gt_3,146,111). coords(mc1gt_3,80,66,111).  visible(mc1gt_3,111).
orientation(mc1gt_2,144,112). coords(mc1gt_2,77,75,112).  visible(mc1gt_2,112). orientation(mc1gt_3,146,112). coords(mc1gt_3,79,67,112).  visible(mc1gt_3,112).
orientation(mc1gt_2,144,113). coords(mc1gt_2,76,75,113).  visible(mc1gt_2,113). orientation(mc1gt_3,146,113). coords(mc1gt_3,79,67,113).  visible(mc1gt_3,113).
orientation(mc1gt_2,144,114). coords(mc1gt_2,76,76,114).  visible(mc1gt_2,114). orientation(mc1gt_3,146,114). coords(mc1gt_3,80,67,114).  visible(mc1gt_3,114).
orientation(mc1gt_2,144,115). coords(mc1gt_2,76,77,115).  visible(mc1gt_2,115). orientation(mc1gt_3,146,115). coords(mc1gt_3,79,67,115).  visible(mc1gt_3,115).
orientation(mc1gt_2,144,116). coords(mc1gt_2,77,78,116).  visible(mc1gt_2,116). orientation(mc1gt_3,146,116). coords(mc1gt_3,80,68,116).  visible(mc1gt_3,116).
orientation(mc1gt_2,144,117). coords(mc1gt_2,77,78,117).  visible(mc1gt_2,117). orientation(mc1gt_3,146,117). coords(mc1gt_3,80,69,117).  visible(mc1gt_3,117).
orientation(mc1gt_2,144,118). coords(mc1gt_2,77,79,118).  visible(mc1gt_2,118). orientation(mc1gt_3,146,118). coords(mc1gt_3,80,70,118).  visible(mc1gt_3,118).
orientation(mc1gt_2,144,119). coords(mc1gt_2,77,79,119).  visible(mc1gt_2,119). orientation(mc1gt_3,146,119). coords(mc1gt_3,81,71,119).  visible(mc1gt_3,119).
orientation(mc1gt_2,144,120). coords(mc1gt_2,77,79,120).  visible(mc1gt_2,120). orientation(mc1gt_3,146,120). coords(mc1gt_3,80,71,120).  visible(mc1gt_3,120).
orientation(mc1gt_2,144,121). coords(mc1gt_2,77,80,121).  visible(mc1gt_2,121). orientation(mc1gt_3,146,121). coords(mc1gt_3,81,71,121).  visible(mc1gt_3,121).
orientation(mc1gt_2,144,122). coords(mc1gt_2,77,80,122).  visible(mc1gt_2,122). orientation(mc1gt_3,146,122). coords(mc1gt_3,83,73,122).  visible(mc1gt_3,122).
orientation(mc1gt_2,144,123). coords(mc1gt_2,77,81,123).  visible(mc1gt_2,123). orientation(mc1gt_3,146,123). coords(mc1gt_3,83,73,123).  visible(mc1gt_3,123).
orientation(mc1gt_2,144,124). coords(mc1gt_2,77,83,124).  visible(mc1gt_2,124). orientation(mc1gt_3,146,124). coords(mc1gt_3,83,73,124).  visible(mc1gt_3,124).
orientation(mc1gt_2,144,125). coords(mc1gt_2,76,85,125).  visible(mc1gt_2,125). orientation(mc1gt_3,146,125). coords(mc1gt_3,83,73,125).  visible(mc1gt_3,125).
orientation(mc1gt_2,144,126). coords(mc1gt_2,76,86,126).  visible(mc1gt_2,126). orientation(mc1gt_3,146,126). coords(mc1gt_3,84,74,126).  visible(mc1gt_3,126).
orientation(mc1gt_2,144,127). coords(mc1gt_2,76,86,127).  visible(mc1gt_2,127). orientation(mc1gt_3,146,127). coords(mc1gt_3,84,74,127).  visible(mc1gt_3,127).
orientation(mc1gt_2,144,128). coords(mc1gt_2,76,87,128).  visible(mc1gt_2,128). orientation(mc1gt_3,146,128). coords(mc1gt_3,84,74,128).  visible(mc1gt_3,128).
orientation(mc1gt_2,144,129). coords(mc1gt_2,76,87,129).  visible(mc1gt_2,129). orientation(mc1gt_3,146,129). coords(mc1gt_3,84,75,129).  visible(mc1gt_3,129).
orientation(mc1gt_2,144,130). coords(mc1gt_2,76,88,130).  visible(mc1gt_2,130). orientation(mc1gt_3,146,130). coords(mc1gt_3,85,76,130).  visible(mc1gt_3,130).
orientation(mc1gt_2,144,131). coords(mc1gt_2,76,88,131).  visible(mc1gt_2,131). orientation(mc1gt_3,146,131). coords(mc1gt_3,85,77,131).  visible(mc1gt_3,131).
orientation(mc1gt_2,144,132). coords(mc1gt_2,76,88,132).  visible(mc1gt_2,132). orientation(mc1gt_3,146,132). coords(mc1gt_3,86,77,132).  visible(mc1gt_3,132).
orientation(mc1gt_2,144,133). coords(mc1gt_2,76,88,133).  visible(mc1gt_2,133). orientation(mc1gt_3,146,133). coords(mc1gt_3,86,77,133).  visible(mc1gt_3,133).
happensAt(disappear(mc1gt_2),132). happensAt(disappear(mc1gt_3),132).
1{simpleEvent(T,mc1gt_2,active); simpleEvent(T,mc1gt_2,inactive); simpleEvent(T,mc1gt_2,walking); simpleEvent(T,mc1gt_2,running)}1 :- T=108..132.
1{simpleEvent(T,mc1gt_3,active); simpleEvent(T,mc1gt_3,inactive); simpleEvent(T,mc1gt_3,walking); simpleEvent(T,mc1gt_3,running)}1 :- T=108..132.
:- holdsAt(moving(mc1gt_2,mc1gt_3),108). :- holdsAt(moving(mc1gt_3,mc1gt_2),108).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),109). :- not holdsAt(moving(mc1gt_3,mc1gt_2),109).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),110). :- not holdsAt(moving(mc1gt_3,mc1gt_2),110).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),111). :- not holdsAt(moving(mc1gt_3,mc1gt_2),111).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),112). :- not holdsAt(moving(mc1gt_3,mc1gt_2),112).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),113). :- not holdsAt(moving(mc1gt_3,mc1gt_2),113).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),114). :- not holdsAt(moving(mc1gt_3,mc1gt_2),114).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),115). :- not holdsAt(moving(mc1gt_3,mc1gt_2),115).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),116). :- not holdsAt(moving(mc1gt_3,mc1gt_2),116).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),117). :- not holdsAt(moving(mc1gt_3,mc1gt_2),117).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),118). :- not holdsAt(moving(mc1gt_3,mc1gt_2),118).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),119). :- not holdsAt(moving(mc1gt_3,mc1gt_2),119).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),120). :- not holdsAt(moving(mc1gt_3,mc1gt_2),120).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),121). :- not holdsAt(moving(mc1gt_3,mc1gt_2),121).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),122). :- not holdsAt(moving(mc1gt_3,mc1gt_2),122).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),123). :- not holdsAt(moving(mc1gt_3,mc1gt_2),123).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),124). :- not holdsAt(moving(mc1gt_3,mc1gt_2),124).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),125). :- not holdsAt(moving(mc1gt_3,mc1gt_2),125).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),126). :- not holdsAt(moving(mc1gt_3,mc1gt_2),126).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),127). :- not holdsAt(moving(mc1gt_3,mc1gt_2),127).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),128). :- not holdsAt(moving(mc1gt_3,mc1gt_2),128).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),129). :- not holdsAt(moving(mc1gt_3,mc1gt_2),129).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),130). :- not holdsAt(moving(mc1gt_3,mc1gt_2),130).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),131). :- not holdsAt(moving(mc1gt_3,mc1gt_2),131).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),132). :- not holdsAt(moving(mc1gt_3,mc1gt_2),132).
:- holdsAt(moving(mc1gt_2,mc1gt_3),133). :- holdsAt(moving(mc1gt_3,mc1gt_2),133).
:- simpleEvent(T,P,inactive), T>108 , T<132.
:- simpleEvent(T,P,running), T>108 , T<132.

#show simpleEvent/3.