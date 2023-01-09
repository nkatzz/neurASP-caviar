
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

	orientation(mws1gt_1,155,104). coords(mws1gt_1,96,174,104).  visible(mws1gt_1,104). happensAt(walking(mws1gt_1),104). orientation(mws1gt_3,174,104). coords(mws1gt_3,238,221,104).  visible(mws1gt_3,104). happensAt(walking(mws1gt_3),104).
orientation(mws1gt_1,155,105). coords(mws1gt_1,98,173,105).  visible(mws1gt_1,105). happensAt(walking(mws1gt_1),105). orientation(mws1gt_3,176,105). coords(mws1gt_3,237,221,105).  visible(mws1gt_3,105). happensAt(walking(mws1gt_3),105).
orientation(mws1gt_1,155,106). coords(mws1gt_1,98,173,106).  visible(mws1gt_1,106). happensAt(walking(mws1gt_1),106). orientation(mws1gt_3,169,106). coords(mws1gt_3,236,221,106).  visible(mws1gt_3,106). happensAt(walking(mws1gt_3),106).
orientation(mws1gt_1,155,107). coords(mws1gt_1,99,173,107).  visible(mws1gt_1,107). happensAt(walking(mws1gt_1),107). orientation(mws1gt_3,172,107). coords(mws1gt_3,235,220,107).  visible(mws1gt_3,107). happensAt(walking(mws1gt_3),107).
orientation(mws1gt_1,155,108). coords(mws1gt_1,100,173,108).  visible(mws1gt_1,108). happensAt(walking(mws1gt_1),108). orientation(mws1gt_3,176,108). coords(mws1gt_3,234,218,108).  visible(mws1gt_3,108). happensAt(walking(mws1gt_3),108).
orientation(mws1gt_1,155,109). coords(mws1gt_1,102,172,109).  visible(mws1gt_1,109). happensAt(walking(mws1gt_1),109). orientation(mws1gt_3,173,109). coords(mws1gt_3,232,216,109).  visible(mws1gt_3,109). happensAt(walking(mws1gt_3),109).
orientation(mws1gt_1,155,110). coords(mws1gt_1,102,172,110).  visible(mws1gt_1,110). happensAt(walking(mws1gt_1),110). orientation(mws1gt_3,170,110). coords(mws1gt_3,231,214,110).  visible(mws1gt_3,110). happensAt(walking(mws1gt_3),110).
orientation(mws1gt_1,155,111). coords(mws1gt_1,102,172,111).  visible(mws1gt_1,111). happensAt(walking(mws1gt_1),111). orientation(mws1gt_3,178,111). coords(mws1gt_3,228,211,111).  visible(mws1gt_3,111). happensAt(walking(mws1gt_3),111).
orientation(mws1gt_1,155,112). coords(mws1gt_1,102,171,112).  visible(mws1gt_1,112). happensAt(walking(mws1gt_1),112). orientation(mws1gt_3,178,112). coords(mws1gt_3,226,209,112).  visible(mws1gt_3,112). happensAt(walking(mws1gt_3),112).
orientation(mws1gt_1,155,113). coords(mws1gt_1,102,171,113).  visible(mws1gt_1,113). happensAt(walking(mws1gt_1),113). orientation(mws1gt_3,176,113). coords(mws1gt_3,224,206,113).  visible(mws1gt_3,113). happensAt(walking(mws1gt_3),113).
orientation(mws1gt_1,155,114). coords(mws1gt_1,102,171,114).  visible(mws1gt_1,114). happensAt(walking(mws1gt_1),114). orientation(mws1gt_3,178,114). coords(mws1gt_3,222,205,114).  visible(mws1gt_3,114). happensAt(walking(mws1gt_3),114).
orientation(mws1gt_1,155,115). coords(mws1gt_1,102,170,115).  visible(mws1gt_1,115). happensAt(walking(mws1gt_1),115). orientation(mws1gt_3,170,115). coords(mws1gt_3,221,203,115).  visible(mws1gt_3,115). happensAt(walking(mws1gt_3),115).
orientation(mws1gt_1,155,116). coords(mws1gt_1,103,169,116).  visible(mws1gt_1,116). happensAt(walking(mws1gt_1),116). orientation(mws1gt_3,160,116). coords(mws1gt_3,220,201,116).  visible(mws1gt_3,116). happensAt(walking(mws1gt_3),116).
orientation(mws1gt_1,155,117). coords(mws1gt_1,105,169,117).  visible(mws1gt_1,117). happensAt(walking(mws1gt_1),117). orientation(mws1gt_3,158,117). coords(mws1gt_3,219,199,117).  visible(mws1gt_3,117). happensAt(walking(mws1gt_3),117).
orientation(mws1gt_1,155,118). coords(mws1gt_1,107,169,118).  visible(mws1gt_1,118). happensAt(walking(mws1gt_1),118). orientation(mws1gt_3,160,118). coords(mws1gt_3,219,198,118).  visible(mws1gt_3,118). happensAt(walking(mws1gt_3),118).
orientation(mws1gt_1,155,119). coords(mws1gt_1,108,168,119).  visible(mws1gt_1,119). happensAt(walking(mws1gt_1),119). orientation(mws1gt_3,157,119). coords(mws1gt_3,219,197,119).  visible(mws1gt_3,119). happensAt(walking(mws1gt_3),119).
orientation(mws1gt_1,155,120). coords(mws1gt_1,110,168,120).  visible(mws1gt_1,120). happensAt(walking(mws1gt_1),120). orientation(mws1gt_3,150,120). coords(mws1gt_3,219,197,120).  visible(mws1gt_3,120). happensAt(walking(mws1gt_3),120).
orientation(mws1gt_1,155,121). coords(mws1gt_1,111,167,121).  visible(mws1gt_1,121). happensAt(walking(mws1gt_1),121). orientation(mws1gt_3,150,121). coords(mws1gt_3,218,197,121).  visible(mws1gt_3,121). happensAt(walking(mws1gt_3),121).
orientation(mws1gt_1,155,122). coords(mws1gt_1,112,166,122).  visible(mws1gt_1,122). happensAt(walking(mws1gt_1),122). orientation(mws1gt_3,150,122). coords(mws1gt_3,216,196,122).  visible(mws1gt_3,122). happensAt(walking(mws1gt_3),122).
orientation(mws1gt_1,155,123). coords(mws1gt_1,113,165,123).  visible(mws1gt_1,123). happensAt(walking(mws1gt_1),123). orientation(mws1gt_3,150,123). coords(mws1gt_3,214,194,123).  visible(mws1gt_3,123). happensAt(walking(mws1gt_3),123).
orientation(mws1gt_1,155,124). coords(mws1gt_1,113,164,124).  visible(mws1gt_1,124). happensAt(walking(mws1gt_1),124). orientation(mws1gt_3,150,124). coords(mws1gt_3,212,191,124).  visible(mws1gt_3,124). happensAt(walking(mws1gt_3),124).
orientation(mws1gt_1,155,125). coords(mws1gt_1,114,163,125).  visible(mws1gt_1,125). happensAt(walking(mws1gt_1),125). orientation(mws1gt_3,150,125). coords(mws1gt_3,210,189,125).  visible(mws1gt_3,125). happensAt(walking(mws1gt_3),125).
orientation(mws1gt_1,155,126). coords(mws1gt_1,115,162,126).  visible(mws1gt_1,126). happensAt(walking(mws1gt_1),126). orientation(mws1gt_3,150,126). coords(mws1gt_3,208,187,126).  visible(mws1gt_3,126). happensAt(walking(mws1gt_3),126).
orientation(mws1gt_1,155,127). coords(mws1gt_1,115,162,127).  visible(mws1gt_1,127). happensAt(walking(mws1gt_1),127). orientation(mws1gt_3,150,127). coords(mws1gt_3,207,184,127).  visible(mws1gt_3,127). happensAt(walking(mws1gt_3),127).
orientation(mws1gt_1,155,128). coords(mws1gt_1,115,161,128).  visible(mws1gt_1,128). happensAt(walking(mws1gt_1),128). orientation(mws1gt_3,150,128). coords(mws1gt_3,206,182,128).  visible(mws1gt_3,128). happensAt(walking(mws1gt_3),128).
#show holdsAt/2.