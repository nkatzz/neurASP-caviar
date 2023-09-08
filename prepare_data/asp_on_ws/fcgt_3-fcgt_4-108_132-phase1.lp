
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


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(fcgt_3,159,108). coords(fcgt_3,103,163,108).  visible(fcgt_3,108). happensAt(inactive(fcgt_3),108). orientation(fcgt_4,165,108). coords(fcgt_4,101,210,108).  visible(fcgt_4,108). happensAt(inactive(fcgt_4),108).
orientation(fcgt_3,159,109). coords(fcgt_3,103,163,109).  visible(fcgt_3,109). happensAt(inactive(fcgt_3),109). orientation(fcgt_4,165,109). coords(fcgt_4,101,210,109).  visible(fcgt_4,109). happensAt(inactive(fcgt_4),109).
orientation(fcgt_3,159,110). coords(fcgt_3,103,163,110).  visible(fcgt_3,110). happensAt(inactive(fcgt_3),110). orientation(fcgt_4,165,110). coords(fcgt_4,101,210,110).  visible(fcgt_4,110). happensAt(inactive(fcgt_4),110).
orientation(fcgt_3,159,111). coords(fcgt_3,103,163,111).  visible(fcgt_3,111). happensAt(inactive(fcgt_3),111). orientation(fcgt_4,165,111). coords(fcgt_4,101,210,111).  visible(fcgt_4,111). happensAt(inactive(fcgt_4),111).
orientation(fcgt_3,159,112). coords(fcgt_3,103,163,112).  visible(fcgt_3,112). happensAt(inactive(fcgt_3),112). orientation(fcgt_4,165,112). coords(fcgt_4,101,210,112).  visible(fcgt_4,112). happensAt(inactive(fcgt_4),112).
orientation(fcgt_3,159,113). coords(fcgt_3,103,163,113).  visible(fcgt_3,113). happensAt(inactive(fcgt_3),113). orientation(fcgt_4,165,113). coords(fcgt_4,101,210,113).  visible(fcgt_4,113). happensAt(inactive(fcgt_4),113).
orientation(fcgt_3,159,114). coords(fcgt_3,103,163,114).  visible(fcgt_3,114). happensAt(inactive(fcgt_3),114). orientation(fcgt_4,165,114). coords(fcgt_4,101,210,114).  visible(fcgt_4,114). happensAt(inactive(fcgt_4),114).
orientation(fcgt_3,159,115). coords(fcgt_3,103,163,115).  visible(fcgt_3,115). happensAt(inactive(fcgt_3),115). orientation(fcgt_4,165,115). coords(fcgt_4,101,210,115).  visible(fcgt_4,115). happensAt(inactive(fcgt_4),115).
orientation(fcgt_3,159,116). coords(fcgt_3,103,163,116).  visible(fcgt_3,116). happensAt(inactive(fcgt_3),116). orientation(fcgt_4,165,116). coords(fcgt_4,101,210,116).  visible(fcgt_4,116). happensAt(inactive(fcgt_4),116).
orientation(fcgt_3,159,117). coords(fcgt_3,103,163,117).  visible(fcgt_3,117). happensAt(inactive(fcgt_3),117). orientation(fcgt_4,165,117). coords(fcgt_4,101,210,117).  visible(fcgt_4,117). happensAt(inactive(fcgt_4),117).
orientation(fcgt_3,159,118). coords(fcgt_3,103,163,118).  visible(fcgt_3,118). happensAt(inactive(fcgt_3),118). orientation(fcgt_4,165,118). coords(fcgt_4,101,210,118).  visible(fcgt_4,118). happensAt(inactive(fcgt_4),118).
orientation(fcgt_3,159,119). coords(fcgt_3,103,163,119).  visible(fcgt_3,119). happensAt(inactive(fcgt_3),119). orientation(fcgt_4,165,119). coords(fcgt_4,101,210,119).  visible(fcgt_4,119). happensAt(inactive(fcgt_4),119).
orientation(fcgt_3,159,120). coords(fcgt_3,103,163,120).  visible(fcgt_3,120). happensAt(inactive(fcgt_3),120). orientation(fcgt_4,165,120). coords(fcgt_4,101,210,120).  visible(fcgt_4,120). happensAt(inactive(fcgt_4),120).
orientation(fcgt_3,159,121). coords(fcgt_3,103,163,121).  visible(fcgt_3,121). happensAt(inactive(fcgt_3),121). orientation(fcgt_4,165,121). coords(fcgt_4,101,210,121).  visible(fcgt_4,121). happensAt(inactive(fcgt_4),121).
orientation(fcgt_3,159,122). coords(fcgt_3,103,163,122).  visible(fcgt_3,122). happensAt(inactive(fcgt_3),122). orientation(fcgt_4,165,122). coords(fcgt_4,101,210,122).  visible(fcgt_4,122). happensAt(inactive(fcgt_4),122).
orientation(fcgt_3,159,123). coords(fcgt_3,103,163,123).  visible(fcgt_3,123). happensAt(inactive(fcgt_3),123). orientation(fcgt_4,165,123). coords(fcgt_4,101,210,123).  visible(fcgt_4,123). happensAt(inactive(fcgt_4),123).
orientation(fcgt_3,159,124). coords(fcgt_3,103,163,124).  visible(fcgt_3,124). happensAt(inactive(fcgt_3),124). orientation(fcgt_4,165,124). coords(fcgt_4,101,210,124).  visible(fcgt_4,124). happensAt(inactive(fcgt_4),124).
orientation(fcgt_3,159,125). coords(fcgt_3,103,163,125).  visible(fcgt_3,125). happensAt(inactive(fcgt_3),125). orientation(fcgt_4,165,125). coords(fcgt_4,101,210,125).  visible(fcgt_4,125). happensAt(inactive(fcgt_4),125).
orientation(fcgt_3,159,126). coords(fcgt_3,103,163,126).  visible(fcgt_3,126). happensAt(inactive(fcgt_3),126). orientation(fcgt_4,165,126). coords(fcgt_4,101,210,126).  visible(fcgt_4,126). happensAt(inactive(fcgt_4),126).
orientation(fcgt_3,159,127). coords(fcgt_3,103,163,127).  visible(fcgt_3,127). happensAt(inactive(fcgt_3),127). orientation(fcgt_4,165,127). coords(fcgt_4,101,210,127).  visible(fcgt_4,127). happensAt(inactive(fcgt_4),127).
orientation(fcgt_3,159,128). coords(fcgt_3,103,163,128).  visible(fcgt_3,128). happensAt(inactive(fcgt_3),128). orientation(fcgt_4,165,128). coords(fcgt_4,101,210,128).  visible(fcgt_4,128). happensAt(inactive(fcgt_4),128).
orientation(fcgt_3,159,129). coords(fcgt_3,103,163,129).  visible(fcgt_3,129). happensAt(inactive(fcgt_3),129). orientation(fcgt_4,165,129). coords(fcgt_4,101,210,129).  visible(fcgt_4,129). happensAt(inactive(fcgt_4),129).
orientation(fcgt_3,159,130). coords(fcgt_3,103,163,130).  visible(fcgt_3,130). happensAt(inactive(fcgt_3),130). orientation(fcgt_4,165,130). coords(fcgt_4,101,210,130).  visible(fcgt_4,130). happensAt(inactive(fcgt_4),130).
orientation(fcgt_3,159,131). coords(fcgt_3,103,163,131).  visible(fcgt_3,131). happensAt(inactive(fcgt_3),131). orientation(fcgt_4,165,131). coords(fcgt_4,101,210,131).  visible(fcgt_4,131). happensAt(inactive(fcgt_4),131).
orientation(fcgt_3,159,132). coords(fcgt_3,103,163,132).  visible(fcgt_3,132). happensAt(inactive(fcgt_3),132). orientation(fcgt_4,165,132). coords(fcgt_4,101,210,132).  visible(fcgt_4,132). happensAt(inactive(fcgt_4),132).
#show holdsAt/2.