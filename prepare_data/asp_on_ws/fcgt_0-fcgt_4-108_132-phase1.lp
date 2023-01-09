
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

	orientation(fcgt_0,175,108). coords(fcgt_0,240,206,108).  visible(fcgt_0,108). happensAt(walking(fcgt_0),108). orientation(fcgt_4,165,108). coords(fcgt_4,101,210,108).  visible(fcgt_4,108). happensAt(inactive(fcgt_4),108).
orientation(fcgt_0,173,109). coords(fcgt_0,241,207,109).  visible(fcgt_0,109). happensAt(walking(fcgt_0),109). orientation(fcgt_4,165,109). coords(fcgt_4,101,210,109).  visible(fcgt_4,109). happensAt(inactive(fcgt_4),109).
orientation(fcgt_0,176,110). coords(fcgt_0,242,209,110).  visible(fcgt_0,110). happensAt(walking(fcgt_0),110). orientation(fcgt_4,165,110). coords(fcgt_4,101,210,110).  visible(fcgt_4,110). happensAt(inactive(fcgt_4),110).
orientation(fcgt_0,178,111). coords(fcgt_0,243,211,111).  visible(fcgt_0,111). happensAt(walking(fcgt_0),111). orientation(fcgt_4,165,111). coords(fcgt_4,101,210,111).  visible(fcgt_4,111). happensAt(inactive(fcgt_4),111).
orientation(fcgt_0,0,112). coords(fcgt_0,244,216,112).  visible(fcgt_0,112). happensAt(walking(fcgt_0),112). orientation(fcgt_4,165,112). coords(fcgt_4,101,210,112).  visible(fcgt_4,112). happensAt(inactive(fcgt_4),112).
orientation(fcgt_0,174,113). coords(fcgt_0,245,220,113).  visible(fcgt_0,113). happensAt(walking(fcgt_0),113). orientation(fcgt_4,165,113). coords(fcgt_4,101,210,113).  visible(fcgt_4,113). happensAt(inactive(fcgt_4),113).
orientation(fcgt_0,170,114). coords(fcgt_0,245,223,114).  visible(fcgt_0,114). happensAt(walking(fcgt_0),114). orientation(fcgt_4,165,114). coords(fcgt_4,101,210,114).  visible(fcgt_4,114). happensAt(inactive(fcgt_4),114).
orientation(fcgt_0,164,115). coords(fcgt_0,246,227,115).  visible(fcgt_0,115). happensAt(walking(fcgt_0),115). orientation(fcgt_4,165,115). coords(fcgt_4,101,210,115).  visible(fcgt_4,115). happensAt(inactive(fcgt_4),115).
orientation(fcgt_0,160,116). coords(fcgt_0,247,229,116).  visible(fcgt_0,116). happensAt(walking(fcgt_0),116). orientation(fcgt_4,165,116). coords(fcgt_4,101,210,116).  visible(fcgt_4,116). happensAt(inactive(fcgt_4),116).
orientation(fcgt_0,163,117). coords(fcgt_0,248,231,117).  visible(fcgt_0,117). happensAt(walking(fcgt_0),117). orientation(fcgt_4,165,117). coords(fcgt_4,101,210,117).  visible(fcgt_4,117). happensAt(inactive(fcgt_4),117).
orientation(fcgt_0,158,118). coords(fcgt_0,249,233,118).  visible(fcgt_0,118). happensAt(walking(fcgt_0),118). orientation(fcgt_4,165,118). coords(fcgt_4,101,210,118).  visible(fcgt_4,118). happensAt(inactive(fcgt_4),118).
orientation(fcgt_0,151,119). coords(fcgt_0,250,236,119).  visible(fcgt_0,119). happensAt(walking(fcgt_0),119). orientation(fcgt_4,165,119). coords(fcgt_4,101,210,119).  visible(fcgt_4,119). happensAt(inactive(fcgt_4),119).
orientation(fcgt_0,152,120). coords(fcgt_0,253,236,120).  visible(fcgt_0,120). happensAt(walking(fcgt_0),120). orientation(fcgt_4,165,120). coords(fcgt_4,101,210,120).  visible(fcgt_4,120). happensAt(inactive(fcgt_4),120).
orientation(fcgt_0,152,121). coords(fcgt_0,255,238,121).  visible(fcgt_0,121). happensAt(walking(fcgt_0),121). orientation(fcgt_4,165,121). coords(fcgt_4,101,210,121).  visible(fcgt_4,121). happensAt(inactive(fcgt_4),121).
orientation(fcgt_0,152,122). coords(fcgt_0,258,239,122).  visible(fcgt_0,122). happensAt(walking(fcgt_0),122). orientation(fcgt_4,165,122). coords(fcgt_4,101,210,122).  visible(fcgt_4,122). happensAt(inactive(fcgt_4),122).
orientation(fcgt_0,152,123). coords(fcgt_0,259,240,123).  visible(fcgt_0,123). happensAt(walking(fcgt_0),123). orientation(fcgt_4,165,123). coords(fcgt_4,101,210,123).  visible(fcgt_4,123). happensAt(inactive(fcgt_4),123).
orientation(fcgt_0,152,124). coords(fcgt_0,261,243,124).  visible(fcgt_0,124). happensAt(walking(fcgt_0),124). orientation(fcgt_4,165,124). coords(fcgt_4,101,210,124).  visible(fcgt_4,124). happensAt(inactive(fcgt_4),124).
orientation(fcgt_0,152,125). coords(fcgt_0,262,245,125).  visible(fcgt_0,125). happensAt(walking(fcgt_0),125). orientation(fcgt_4,165,125). coords(fcgt_4,101,210,125).  visible(fcgt_4,125). happensAt(inactive(fcgt_4),125).
orientation(fcgt_0,152,126). coords(fcgt_0,263,249,126).  visible(fcgt_0,126). happensAt(walking(fcgt_0),126). orientation(fcgt_4,165,126). coords(fcgt_4,101,210,126).  visible(fcgt_4,126). happensAt(inactive(fcgt_4),126).
orientation(fcgt_0,152,127). coords(fcgt_0,264,252,127).  visible(fcgt_0,127). happensAt(walking(fcgt_0),127). orientation(fcgt_4,165,127). coords(fcgt_4,101,210,127).  visible(fcgt_4,127). happensAt(inactive(fcgt_4),127).
orientation(fcgt_0,152,128). coords(fcgt_0,264,255,128).  visible(fcgt_0,128). happensAt(walking(fcgt_0),128). orientation(fcgt_4,165,128). coords(fcgt_4,101,210,128).  visible(fcgt_4,128). happensAt(inactive(fcgt_4),128).
orientation(fcgt_0,152,129). coords(fcgt_0,267,259,129).  visible(fcgt_0,129). happensAt(walking(fcgt_0),129). orientation(fcgt_4,165,129). coords(fcgt_4,101,210,129).  visible(fcgt_4,129). happensAt(inactive(fcgt_4),129).
orientation(fcgt_0,152,130). coords(fcgt_0,269,263,130).  visible(fcgt_0,130). happensAt(walking(fcgt_0),130). orientation(fcgt_4,165,130). coords(fcgt_4,101,210,130).  visible(fcgt_4,130). happensAt(inactive(fcgt_4),130).
orientation(fcgt_0,152,131). coords(fcgt_0,272,266,131).  visible(fcgt_0,131). happensAt(walking(fcgt_0),131). orientation(fcgt_4,165,131). coords(fcgt_4,101,210,131).  visible(fcgt_4,131). happensAt(inactive(fcgt_4),131).
orientation(fcgt_0,152,132). coords(fcgt_0,277,268,132).  visible(fcgt_0,132). happensAt(walking(fcgt_0),132). orientation(fcgt_4,165,132). coords(fcgt_4,101,210,132).  visible(fcgt_4,132). happensAt(inactive(fcgt_4),132).
#show holdsAt/2.