
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

	orientation(fra2gt_0,168,108). coords(fra2gt_0,215,181,108).  visible(fra2gt_0,108). happensAt(walking(fra2gt_0),108). orientation(fra2gt_5,150,108). coords(fra2gt_5,29,124,108).  visible(fra2gt_5,108). happensAt(walking(fra2gt_5),108).
orientation(fra2gt_0,168,109). coords(fra2gt_0,219,183,109).  visible(fra2gt_0,109). happensAt(walking(fra2gt_0),109). orientation(fra2gt_5,150,109). coords(fra2gt_5,29,124,109).  visible(fra2gt_5,109). happensAt(walking(fra2gt_5),109).
orientation(fra2gt_0,168,110). coords(fra2gt_0,221,186,110).  visible(fra2gt_0,110). happensAt(walking(fra2gt_0),110). orientation(fra2gt_5,150,110). coords(fra2gt_5,27,125,110).  visible(fra2gt_5,110). happensAt(walking(fra2gt_5),110).
orientation(fra2gt_0,168,111). coords(fra2gt_0,224,189,111).  visible(fra2gt_0,111). happensAt(walking(fra2gt_0),111). orientation(fra2gt_5,150,111). coords(fra2gt_5,27,125,111).  visible(fra2gt_5,111). happensAt(walking(fra2gt_5),111).
orientation(fra2gt_0,168,112). coords(fra2gt_0,225,191,112).  visible(fra2gt_0,112). happensAt(walking(fra2gt_0),112). orientation(fra2gt_5,150,112). coords(fra2gt_5,27,125,112).  visible(fra2gt_5,112). happensAt(walking(fra2gt_5),112).
orientation(fra2gt_0,168,113). coords(fra2gt_0,227,193,113).  visible(fra2gt_0,113). happensAt(walking(fra2gt_0),113). orientation(fra2gt_5,150,113). coords(fra2gt_5,27,125,113).  visible(fra2gt_5,113). happensAt(walking(fra2gt_5),113).
orientation(fra2gt_0,168,114). coords(fra2gt_0,229,196,114).  visible(fra2gt_0,114). happensAt(walking(fra2gt_0),114). orientation(fra2gt_5,150,114). coords(fra2gt_5,27,126,114).  visible(fra2gt_5,114). happensAt(walking(fra2gt_5),114).
orientation(fra2gt_0,168,115). coords(fra2gt_0,230,200,115).  visible(fra2gt_0,115). happensAt(walking(fra2gt_0),115). orientation(fra2gt_5,150,115). coords(fra2gt_5,27,126,115).  visible(fra2gt_5,115). happensAt(walking(fra2gt_5),115).
orientation(fra2gt_0,168,116). coords(fra2gt_0,231,203,116).  visible(fra2gt_0,116). happensAt(walking(fra2gt_0),116). orientation(fra2gt_5,150,116). coords(fra2gt_5,27,126,116).  visible(fra2gt_5,116). happensAt(walking(fra2gt_5),116).
orientation(fra2gt_0,175,117). coords(fra2gt_0,232,207,117).  visible(fra2gt_0,117). happensAt(walking(fra2gt_0),117). orientation(fra2gt_5,150,117). coords(fra2gt_5,27,126,117).  visible(fra2gt_5,117). happensAt(walking(fra2gt_5),117).
orientation(fra2gt_0,9,118). coords(fra2gt_0,233,211,118).  visible(fra2gt_0,118). happensAt(walking(fra2gt_0),118). orientation(fra2gt_5,150,118). coords(fra2gt_5,27,128,118).  visible(fra2gt_5,118). happensAt(walking(fra2gt_5),118).
orientation(fra2gt_0,9,119). coords(fra2gt_0,234,215,119).  visible(fra2gt_0,119). happensAt(walking(fra2gt_0),119). orientation(fra2gt_5,150,119). coords(fra2gt_5,27,128,119).  visible(fra2gt_5,119). happensAt(walking(fra2gt_5),119).
orientation(fra2gt_0,9,120). coords(fra2gt_0,236,218,120).  visible(fra2gt_0,120). happensAt(walking(fra2gt_0),120). orientation(fra2gt_5,150,120). coords(fra2gt_5,26,129,120).  visible(fra2gt_5,120). happensAt(walking(fra2gt_5),120).
orientation(fra2gt_0,9,121). coords(fra2gt_0,239,219,121).  visible(fra2gt_0,121). happensAt(walking(fra2gt_0),121). orientation(fra2gt_5,150,121). coords(fra2gt_5,26,129,121).  visible(fra2gt_5,121). happensAt(walking(fra2gt_5),121).
orientation(fra2gt_0,9,122). coords(fra2gt_0,241,222,122).  visible(fra2gt_0,122). happensAt(walking(fra2gt_0),122). orientation(fra2gt_5,150,122). coords(fra2gt_5,26,129,122).  visible(fra2gt_5,122). happensAt(walking(fra2gt_5),122).
orientation(fra2gt_0,9,123). coords(fra2gt_0,244,224,123).  visible(fra2gt_0,123). happensAt(walking(fra2gt_0),123). orientation(fra2gt_5,150,123). coords(fra2gt_5,26,129,123).  visible(fra2gt_5,123). happensAt(walking(fra2gt_5),123).
orientation(fra2gt_0,9,124). coords(fra2gt_0,246,226,124).  visible(fra2gt_0,124). happensAt(walking(fra2gt_0),124). orientation(fra2gt_5,150,124). coords(fra2gt_5,26,131,124).  visible(fra2gt_5,124). happensAt(walking(fra2gt_5),124).
orientation(fra2gt_0,9,125). coords(fra2gt_0,248,230,125).  visible(fra2gt_0,125). happensAt(walking(fra2gt_0),125). orientation(fra2gt_5,150,125). coords(fra2gt_5,26,131,125).  visible(fra2gt_5,125). happensAt(walking(fra2gt_5),125).
orientation(fra2gt_0,9,126). coords(fra2gt_0,249,232,126).  visible(fra2gt_0,126). happensAt(walking(fra2gt_0),126). orientation(fra2gt_5,150,126). coords(fra2gt_5,26,131,126).  visible(fra2gt_5,126). happensAt(walking(fra2gt_5),126).
orientation(fra2gt_0,10,127). coords(fra2gt_0,251,236,127).  visible(fra2gt_0,127). happensAt(walking(fra2gt_0),127). orientation(fra2gt_5,150,127). coords(fra2gt_5,26,131,127).  visible(fra2gt_5,127). happensAt(walking(fra2gt_5),127).
orientation(fra2gt_0,18,128). coords(fra2gt_0,253,241,128).  visible(fra2gt_0,128). happensAt(walking(fra2gt_0),128). orientation(fra2gt_5,150,128). coords(fra2gt_5,26,131,128).  visible(fra2gt_5,128). happensAt(walking(fra2gt_5),128).
orientation(fra2gt_0,28,129). coords(fra2gt_0,257,246,129).  visible(fra2gt_0,129). happensAt(walking(fra2gt_0),129). orientation(fra2gt_5,150,129). coords(fra2gt_5,26,131,129).  visible(fra2gt_5,129). happensAt(walking(fra2gt_5),129).
orientation(fra2gt_0,28,130). coords(fra2gt_0,259,250,130).  visible(fra2gt_0,130). happensAt(walking(fra2gt_0),130). orientation(fra2gt_5,150,130). coords(fra2gt_5,26,131,130).  visible(fra2gt_5,130). happensAt(walking(fra2gt_5),130).
orientation(fra2gt_0,28,131). coords(fra2gt_0,263,251,131).  visible(fra2gt_0,131). happensAt(walking(fra2gt_0),131). orientation(fra2gt_5,150,131). coords(fra2gt_5,26,131,131).  visible(fra2gt_5,131). happensAt(walking(fra2gt_5),131).
orientation(fra2gt_0,28,132). coords(fra2gt_0,265,254,132).  visible(fra2gt_0,132). happensAt(walking(fra2gt_0),132). orientation(fra2gt_5,150,132). coords(fra2gt_5,26,131,132).  visible(fra2gt_5,132). happensAt(walking(fra2gt_5),132).
#show holdsAt/2.