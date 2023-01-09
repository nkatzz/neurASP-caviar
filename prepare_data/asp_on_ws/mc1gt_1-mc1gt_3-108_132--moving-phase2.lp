

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

	person(mc1gt_1;mc1gt_3).
orientation(mc1gt_1,145,108). coords(mc1gt_1,97,82,108).  visible(mc1gt_1,108). orientation(mc1gt_3,146,108). coords(mc1gt_3,83,65,108).  happensAt(appear(mc1gt_3),108).
orientation(mc1gt_1,145,109). coords(mc1gt_1,98,82,109).  visible(mc1gt_1,109). orientation(mc1gt_3,146,109). coords(mc1gt_3,79,65,109).  visible(mc1gt_3,109).
orientation(mc1gt_1,145,110). coords(mc1gt_1,99,82,110).  visible(mc1gt_1,110). orientation(mc1gt_3,146,110). coords(mc1gt_3,79,66,110).  visible(mc1gt_3,110).
orientation(mc1gt_1,145,111). coords(mc1gt_1,100,82,111).  visible(mc1gt_1,111). orientation(mc1gt_3,146,111). coords(mc1gt_3,80,66,111).  visible(mc1gt_3,111).
orientation(mc1gt_1,145,112). coords(mc1gt_1,100,82,112).  visible(mc1gt_1,112). orientation(mc1gt_3,146,112). coords(mc1gt_3,79,67,112).  visible(mc1gt_3,112).
orientation(mc1gt_1,145,113). coords(mc1gt_1,101,82,113).  visible(mc1gt_1,113). orientation(mc1gt_3,146,113). coords(mc1gt_3,79,67,113).  visible(mc1gt_3,113).
orientation(mc1gt_1,145,114). coords(mc1gt_1,102,82,114).  visible(mc1gt_1,114). orientation(mc1gt_3,146,114). coords(mc1gt_3,80,67,114).  visible(mc1gt_3,114).
orientation(mc1gt_1,145,115). coords(mc1gt_1,102,82,115).  visible(mc1gt_1,115). orientation(mc1gt_3,146,115). coords(mc1gt_3,79,67,115).  visible(mc1gt_3,115).
orientation(mc1gt_1,145,116). coords(mc1gt_1,102,82,116).  visible(mc1gt_1,116). orientation(mc1gt_3,146,116). coords(mc1gt_3,80,68,116).  visible(mc1gt_3,116).
orientation(mc1gt_1,145,117). coords(mc1gt_1,102,82,117).  visible(mc1gt_1,117). orientation(mc1gt_3,146,117). coords(mc1gt_3,80,69,117).  visible(mc1gt_3,117).
orientation(mc1gt_1,145,118). coords(mc1gt_1,102,82,118).  visible(mc1gt_1,118). orientation(mc1gt_3,146,118). coords(mc1gt_3,80,70,118).  visible(mc1gt_3,118).
orientation(mc1gt_1,145,119). coords(mc1gt_1,102,82,119).  visible(mc1gt_1,119). orientation(mc1gt_3,146,119). coords(mc1gt_3,81,71,119).  visible(mc1gt_3,119).
orientation(mc1gt_1,145,120). coords(mc1gt_1,103,82,120).  visible(mc1gt_1,120). orientation(mc1gt_3,146,120). coords(mc1gt_3,80,71,120).  visible(mc1gt_3,120).
orientation(mc1gt_1,145,121). coords(mc1gt_1,103,82,121).  visible(mc1gt_1,121). orientation(mc1gt_3,146,121). coords(mc1gt_3,81,71,121).  visible(mc1gt_3,121).
orientation(mc1gt_1,145,122). coords(mc1gt_1,105,82,122).  visible(mc1gt_1,122). orientation(mc1gt_3,146,122). coords(mc1gt_3,83,73,122).  visible(mc1gt_3,122).
orientation(mc1gt_1,145,123). coords(mc1gt_1,106,82,123).  visible(mc1gt_1,123). orientation(mc1gt_3,146,123). coords(mc1gt_3,83,73,123).  visible(mc1gt_3,123).
orientation(mc1gt_1,145,124). coords(mc1gt_1,107,82,124).  visible(mc1gt_1,124). orientation(mc1gt_3,146,124). coords(mc1gt_3,83,73,124).  visible(mc1gt_3,124).
orientation(mc1gt_1,145,125). coords(mc1gt_1,108,82,125).  visible(mc1gt_1,125). orientation(mc1gt_3,146,125). coords(mc1gt_3,83,73,125).  visible(mc1gt_3,125).
orientation(mc1gt_1,145,126). coords(mc1gt_1,109,82,126).  visible(mc1gt_1,126). orientation(mc1gt_3,146,126). coords(mc1gt_3,84,74,126).  visible(mc1gt_3,126).
orientation(mc1gt_1,145,127). coords(mc1gt_1,110,82,127).  visible(mc1gt_1,127). orientation(mc1gt_3,146,127). coords(mc1gt_3,84,74,127).  visible(mc1gt_3,127).
orientation(mc1gt_1,145,128). coords(mc1gt_1,110,82,128).  visible(mc1gt_1,128). orientation(mc1gt_3,146,128). coords(mc1gt_3,84,74,128).  visible(mc1gt_3,128).
orientation(mc1gt_1,145,129). coords(mc1gt_1,110,82,129).  visible(mc1gt_1,129). orientation(mc1gt_3,146,129). coords(mc1gt_3,84,75,129).  visible(mc1gt_3,129).
orientation(mc1gt_1,145,130). coords(mc1gt_1,111,82,130).  visible(mc1gt_1,130). orientation(mc1gt_3,146,130). coords(mc1gt_3,85,76,130).  visible(mc1gt_3,130).
orientation(mc1gt_1,145,131). coords(mc1gt_1,111,81,131).  visible(mc1gt_1,131). orientation(mc1gt_3,146,131). coords(mc1gt_3,85,77,131).  visible(mc1gt_3,131).
orientation(mc1gt_1,145,132). coords(mc1gt_1,112,81,132).  visible(mc1gt_1,132). orientation(mc1gt_3,146,132). coords(mc1gt_3,86,77,132).  visible(mc1gt_3,132).
orientation(mc1gt_1,145,133). coords(mc1gt_1,112,81,133).  visible(mc1gt_1,133). orientation(mc1gt_3,146,133). coords(mc1gt_3,86,77,133).  visible(mc1gt_3,133).
happensAt(disappear(mc1gt_1),132). happensAt(disappear(mc1gt_3),132).
1{simpleEvent(T,mc1gt_1,active); simpleEvent(T,mc1gt_1,inactive); simpleEvent(T,mc1gt_1,walking); simpleEvent(T,mc1gt_1,running)}1 :- T=108..132.
1{simpleEvent(T,mc1gt_3,active); simpleEvent(T,mc1gt_3,inactive); simpleEvent(T,mc1gt_3,walking); simpleEvent(T,mc1gt_3,running)}1 :- T=108..132.
:- holdsAt(moving(mc1gt_1,mc1gt_3),108). :- holdsAt(moving(mc1gt_3,mc1gt_1),108).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),109). :- not holdsAt(moving(mc1gt_3,mc1gt_1),109).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),110). :- not holdsAt(moving(mc1gt_3,mc1gt_1),110).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),111). :- not holdsAt(moving(mc1gt_3,mc1gt_1),111).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),112). :- not holdsAt(moving(mc1gt_3,mc1gt_1),112).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),113). :- not holdsAt(moving(mc1gt_3,mc1gt_1),113).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),114). :- not holdsAt(moving(mc1gt_3,mc1gt_1),114).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),115). :- not holdsAt(moving(mc1gt_3,mc1gt_1),115).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),116). :- not holdsAt(moving(mc1gt_3,mc1gt_1),116).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),117). :- not holdsAt(moving(mc1gt_3,mc1gt_1),117).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),118). :- not holdsAt(moving(mc1gt_3,mc1gt_1),118).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),119). :- not holdsAt(moving(mc1gt_3,mc1gt_1),119).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),120). :- not holdsAt(moving(mc1gt_3,mc1gt_1),120).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),121). :- not holdsAt(moving(mc1gt_3,mc1gt_1),121).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),122). :- not holdsAt(moving(mc1gt_3,mc1gt_1),122).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),123). :- not holdsAt(moving(mc1gt_3,mc1gt_1),123).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),124). :- not holdsAt(moving(mc1gt_3,mc1gt_1),124).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),125). :- not holdsAt(moving(mc1gt_3,mc1gt_1),125).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),126). :- not holdsAt(moving(mc1gt_3,mc1gt_1),126).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),127). :- not holdsAt(moving(mc1gt_3,mc1gt_1),127).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),128). :- not holdsAt(moving(mc1gt_3,mc1gt_1),128).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),129). :- not holdsAt(moving(mc1gt_3,mc1gt_1),129).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),130). :- not holdsAt(moving(mc1gt_3,mc1gt_1),130).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),131). :- not holdsAt(moving(mc1gt_3,mc1gt_1),131).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),132). :- not holdsAt(moving(mc1gt_3,mc1gt_1),132).
:- holdsAt(moving(mc1gt_1,mc1gt_3),133). :- holdsAt(moving(mc1gt_3,mc1gt_1),133).
:- simpleEvent(T,P,inactive), T>108 , T<132.
:- simpleEvent(T,P,running), T>108 , T<132.

#show simpleEvent/3.