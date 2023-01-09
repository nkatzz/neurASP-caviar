
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

	orientation(br4gt_0,137,105). coords(br4gt_0,264,190,105).  visible(br4gt_0,105). happensAt(walking(br4gt_0),105). orientation(br4gt_1,43,105). coords(br4gt_1,239,281,105).  happensAt(appear(br4gt_1),105). happensAt(walking(br4gt_1),105).
orientation(br4gt_0,135,106). coords(br4gt_0,268,192,106).  visible(br4gt_0,106). happensAt(walking(br4gt_0),106). orientation(br4gt_1,52,106). coords(br4gt_1,239,278,106).  visible(br4gt_1,106). happensAt(walking(br4gt_1),106).
orientation(br4gt_0,132,107). coords(br4gt_0,271,195,107).  visible(br4gt_0,107). happensAt(walking(br4gt_0),107). orientation(br4gt_1,62,107). coords(br4gt_1,237,276,107).  visible(br4gt_1,107). happensAt(walking(br4gt_1),107).
orientation(br4gt_0,130,108). coords(br4gt_0,274,196,108).  visible(br4gt_0,108). happensAt(walking(br4gt_0),108). orientation(br4gt_1,69,108). coords(br4gt_1,236,274,108).  visible(br4gt_1,108). happensAt(walking(br4gt_1),108).
orientation(br4gt_0,128,109). coords(br4gt_0,276,199,109).  visible(br4gt_0,109). happensAt(walking(br4gt_0),109). orientation(br4gt_1,52,109). coords(br4gt_1,229,273,109).  visible(br4gt_1,109). happensAt(walking(br4gt_1),109).
orientation(br4gt_0,126,110). coords(br4gt_0,278,201,110).  visible(br4gt_0,110). happensAt(walking(br4gt_0),110). orientation(br4gt_1,44,110). coords(br4gt_1,227,273,110).  visible(br4gt_1,110). happensAt(walking(br4gt_1),110).
orientation(br4gt_0,124,111). coords(br4gt_0,279,204,111).  visible(br4gt_0,111). happensAt(walking(br4gt_0),111). orientation(br4gt_1,40,111). coords(br4gt_1,227,272,111).  visible(br4gt_1,111). happensAt(walking(br4gt_1),111).
orientation(br4gt_0,122,112). coords(br4gt_0,281,207,112).  visible(br4gt_0,112). happensAt(walking(br4gt_0),112). orientation(br4gt_1,36,112). coords(br4gt_1,226,272,112).  visible(br4gt_1,112). happensAt(walking(br4gt_1),112).
orientation(br4gt_0,120,113). coords(br4gt_0,282,209,113).  visible(br4gt_0,113). happensAt(walking(br4gt_0),113). orientation(br4gt_1,23,113). coords(br4gt_1,226,272,113).  visible(br4gt_1,113). happensAt(walking(br4gt_1),113).
orientation(br4gt_0,118,114). coords(br4gt_0,282,212,114).  visible(br4gt_0,114). happensAt(walking(br4gt_0),114). orientation(br4gt_1,17,114). coords(br4gt_1,226,272,114).  visible(br4gt_1,114). happensAt(walking(br4gt_1),114).
orientation(br4gt_0,116,115). coords(br4gt_0,283,215,115).  visible(br4gt_0,115). happensAt(walking(br4gt_0),115). orientation(br4gt_1,14,115). coords(br4gt_1,226,272,115).  visible(br4gt_1,115). happensAt(walking(br4gt_1),115).
orientation(br4gt_0,117,116). coords(br4gt_0,285,218,116).  visible(br4gt_0,116). happensAt(walking(br4gt_0),116). orientation(br4gt_1,9,116). coords(br4gt_1,227,271,116).  visible(br4gt_1,116). happensAt(walking(br4gt_1),116).
orientation(br4gt_0,121,117). coords(br4gt_0,286,221,117).  visible(br4gt_0,117). happensAt(walking(br4gt_0),117). orientation(br4gt_1,8,117). coords(br4gt_1,226,270,117).  visible(br4gt_1,117). happensAt(walking(br4gt_1),117).
orientation(br4gt_0,125,118). coords(br4gt_0,288,225,118).  visible(br4gt_0,118). happensAt(walking(br4gt_0),118). orientation(br4gt_1,8,118). coords(br4gt_1,226,268,118).  visible(br4gt_1,118). happensAt(walking(br4gt_1),118).
orientation(br4gt_0,132,119). coords(br4gt_0,291,229,119).  visible(br4gt_0,119). happensAt(walking(br4gt_0),119). orientation(br4gt_1,4,119). coords(br4gt_1,225,265,119).  visible(br4gt_1,119). happensAt(walking(br4gt_1),119).
orientation(br4gt_0,129,120). coords(br4gt_0,292,231,120).  visible(br4gt_0,120). happensAt(walking(br4gt_0),120). orientation(br4gt_1,4,120). coords(br4gt_1,225,262,120).  visible(br4gt_1,120). happensAt(walking(br4gt_1),120).
orientation(br4gt_0,126,121). coords(br4gt_0,294,232,121).  visible(br4gt_0,121). happensAt(walking(br4gt_0),121). orientation(br4gt_1,9,121). coords(br4gt_1,224,257,121).  visible(br4gt_1,121). happensAt(walking(br4gt_1),121).
orientation(br4gt_0,125,122). coords(br4gt_0,295,232,122).  visible(br4gt_0,122). happensAt(walking(br4gt_0),122). orientation(br4gt_1,13,122). coords(br4gt_1,223,253,122).  visible(br4gt_1,122). happensAt(walking(br4gt_1),122).
orientation(br4gt_0,126,123). coords(br4gt_0,296,234,123).  visible(br4gt_0,123). happensAt(walking(br4gt_0),123). orientation(br4gt_1,17,123). coords(br4gt_1,222,251,123).  visible(br4gt_1,123). happensAt(walking(br4gt_1),123).
orientation(br4gt_0,123,124). coords(br4gt_0,298,236,124).  visible(br4gt_0,124). happensAt(walking(br4gt_0),124). orientation(br4gt_1,19,124). coords(br4gt_1,221,249,124).  visible(br4gt_1,124). happensAt(walking(br4gt_1),124).
orientation(br4gt_0,115,125). coords(br4gt_0,299,240,125).  visible(br4gt_0,125). happensAt(walking(br4gt_0),125). orientation(br4gt_1,15,125). coords(br4gt_1,221,249,125).  visible(br4gt_1,125). happensAt(walking(br4gt_1),125).
orientation(br4gt_0,118,126). coords(br4gt_0,301,244,126).  visible(br4gt_0,126). happensAt(walking(br4gt_0),126). orientation(br4gt_1,4,126). coords(br4gt_1,221,248,126).  visible(br4gt_1,126). happensAt(walking(br4gt_1),126).
orientation(br4gt_0,112,127). coords(br4gt_0,302,248,127).  visible(br4gt_0,127). happensAt(walking(br4gt_0),127). orientation(br4gt_1,5,127). coords(br4gt_1,220,247,127).  visible(br4gt_1,127). happensAt(walking(br4gt_1),127).
orientation(br4gt_0,114,128). coords(br4gt_0,302,252,128).  visible(br4gt_0,128). happensAt(walking(br4gt_0),128). orientation(br4gt_1,3,128). coords(br4gt_1,219,245,128).  visible(br4gt_1,128). happensAt(walking(br4gt_1),128).
orientation(br4gt_0,123,129). coords(br4gt_0,303,256,129).  visible(br4gt_0,129). happensAt(walking(br4gt_0),129). orientation(br4gt_1,3,129). coords(br4gt_1,219,243,129).  visible(br4gt_1,129). happensAt(walking(br4gt_1),129).
#show holdsAt/2.