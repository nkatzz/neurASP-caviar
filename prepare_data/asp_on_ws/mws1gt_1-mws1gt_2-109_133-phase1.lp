
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

	orientation(mws1gt_1,155,109). coords(mws1gt_1,102,172,109).  visible(mws1gt_1,109). happensAt(walking(mws1gt_1),109). orientation(mws1gt_2,164,109). coords(mws1gt_2,34,197,109).  visible(mws1gt_2,109). happensAt(walking(mws1gt_2),109).
orientation(mws1gt_1,155,110). coords(mws1gt_1,102,172,110).  visible(mws1gt_1,110). happensAt(walking(mws1gt_1),110). orientation(mws1gt_2,164,110). coords(mws1gt_2,34,197,110).  visible(mws1gt_2,110). happensAt(walking(mws1gt_2),110).
orientation(mws1gt_1,155,111). coords(mws1gt_1,102,172,111).  visible(mws1gt_1,111). happensAt(walking(mws1gt_1),111). orientation(mws1gt_2,164,111). coords(mws1gt_2,34,198,111).  visible(mws1gt_2,111). happensAt(walking(mws1gt_2),111).
orientation(mws1gt_1,155,112). coords(mws1gt_1,102,171,112).  visible(mws1gt_1,112). happensAt(walking(mws1gt_1),112). orientation(mws1gt_2,164,112). coords(mws1gt_2,34,198,112).  visible(mws1gt_2,112). happensAt(walking(mws1gt_2),112).
orientation(mws1gt_1,155,113). coords(mws1gt_1,102,171,113).  visible(mws1gt_1,113). happensAt(walking(mws1gt_1),113). orientation(mws1gt_2,164,113). coords(mws1gt_2,34,200,113).  visible(mws1gt_2,113). happensAt(walking(mws1gt_2),113).
orientation(mws1gt_1,155,114). coords(mws1gt_1,102,171,114).  visible(mws1gt_1,114). happensAt(walking(mws1gt_1),114). orientation(mws1gt_2,164,114). coords(mws1gt_2,34,200,114).  visible(mws1gt_2,114). happensAt(walking(mws1gt_2),114).
orientation(mws1gt_1,155,115). coords(mws1gt_1,102,170,115).  visible(mws1gt_1,115). happensAt(walking(mws1gt_1),115). orientation(mws1gt_2,164,115). coords(mws1gt_2,34,201,115).  visible(mws1gt_2,115). happensAt(walking(mws1gt_2),115).
orientation(mws1gt_1,155,116). coords(mws1gt_1,103,169,116).  visible(mws1gt_1,116). happensAt(walking(mws1gt_1),116). orientation(mws1gt_2,164,116). coords(mws1gt_2,34,202,116).  visible(mws1gt_2,116). happensAt(walking(mws1gt_2),116).
orientation(mws1gt_1,155,117). coords(mws1gt_1,105,169,117).  visible(mws1gt_1,117). happensAt(walking(mws1gt_1),117). orientation(mws1gt_2,164,117). coords(mws1gt_2,34,203,117).  visible(mws1gt_2,117). happensAt(walking(mws1gt_2),117).
orientation(mws1gt_1,155,118). coords(mws1gt_1,107,169,118).  visible(mws1gt_1,118). happensAt(walking(mws1gt_1),118). orientation(mws1gt_2,164,118). coords(mws1gt_2,34,203,118).  visible(mws1gt_2,118). happensAt(walking(mws1gt_2),118).
orientation(mws1gt_1,155,119). coords(mws1gt_1,108,168,119).  visible(mws1gt_1,119). happensAt(walking(mws1gt_1),119). orientation(mws1gt_2,164,119). coords(mws1gt_2,34,204,119).  visible(mws1gt_2,119). happensAt(walking(mws1gt_2),119).
orientation(mws1gt_1,155,120). coords(mws1gt_1,110,168,120).  visible(mws1gt_1,120). happensAt(walking(mws1gt_1),120). orientation(mws1gt_2,164,120). coords(mws1gt_2,34,205,120).  visible(mws1gt_2,120). happensAt(walking(mws1gt_2),120).
orientation(mws1gt_1,155,121). coords(mws1gt_1,111,167,121).  visible(mws1gt_1,121). happensAt(walking(mws1gt_1),121). orientation(mws1gt_2,164,121). coords(mws1gt_2,34,205,121).  visible(mws1gt_2,121). happensAt(walking(mws1gt_2),121).
orientation(mws1gt_1,155,122). coords(mws1gt_1,112,166,122).  visible(mws1gt_1,122). happensAt(walking(mws1gt_1),122). orientation(mws1gt_2,164,122). coords(mws1gt_2,33,205,122).  visible(mws1gt_2,122). happensAt(walking(mws1gt_2),122).
orientation(mws1gt_1,155,123). coords(mws1gt_1,113,165,123).  visible(mws1gt_1,123). happensAt(walking(mws1gt_1),123). orientation(mws1gt_2,164,123). coords(mws1gt_2,33,205,123).  visible(mws1gt_2,123). happensAt(walking(mws1gt_2),123).
orientation(mws1gt_1,155,124). coords(mws1gt_1,113,164,124).  visible(mws1gt_1,124). happensAt(walking(mws1gt_1),124). orientation(mws1gt_2,164,124). coords(mws1gt_2,33,206,124).  visible(mws1gt_2,124). happensAt(walking(mws1gt_2),124).
orientation(mws1gt_1,155,125). coords(mws1gt_1,114,163,125).  visible(mws1gt_1,125). happensAt(walking(mws1gt_1),125). orientation(mws1gt_2,164,125). coords(mws1gt_2,32,206,125).  visible(mws1gt_2,125). happensAt(walking(mws1gt_2),125).
orientation(mws1gt_1,155,126). coords(mws1gt_1,115,162,126).  visible(mws1gt_1,126). happensAt(walking(mws1gt_1),126). orientation(mws1gt_2,164,126). coords(mws1gt_2,32,206,126).  visible(mws1gt_2,126). happensAt(walking(mws1gt_2),126).
orientation(mws1gt_1,155,127). coords(mws1gt_1,115,162,127).  visible(mws1gt_1,127). happensAt(walking(mws1gt_1),127). orientation(mws1gt_2,164,127). coords(mws1gt_2,32,207,127).  visible(mws1gt_2,127). happensAt(walking(mws1gt_2),127).
orientation(mws1gt_1,155,128). coords(mws1gt_1,115,161,128).  visible(mws1gt_1,128). happensAt(walking(mws1gt_1),128). orientation(mws1gt_2,164,128). coords(mws1gt_2,32,207,128).  visible(mws1gt_2,128). happensAt(walking(mws1gt_2),128).
orientation(mws1gt_1,155,129). coords(mws1gt_1,115,159,129).  visible(mws1gt_1,129). happensAt(walking(mws1gt_1),129). orientation(mws1gt_2,164,129). coords(mws1gt_2,29,207,129).  visible(mws1gt_2,129). happensAt(walking(mws1gt_2),129).
orientation(mws1gt_1,155,130). coords(mws1gt_1,115,157,130).  visible(mws1gt_1,130). happensAt(walking(mws1gt_1),130). orientation(mws1gt_2,164,130). coords(mws1gt_2,29,208,130).  visible(mws1gt_2,130). happensAt(walking(mws1gt_2),130).
orientation(mws1gt_1,155,131). coords(mws1gt_1,118,156,131).  visible(mws1gt_1,131). happensAt(walking(mws1gt_1),131). orientation(mws1gt_2,164,131). coords(mws1gt_2,29,208,131).  visible(mws1gt_2,131). happensAt(walking(mws1gt_2),131).
orientation(mws1gt_1,155,132). coords(mws1gt_1,120,156,132).  visible(mws1gt_1,132). happensAt(walking(mws1gt_1),132). orientation(mws1gt_2,164,132). coords(mws1gt_2,29,209,132).  visible(mws1gt_2,132). happensAt(walking(mws1gt_2),132).
orientation(mws1gt_1,155,133). coords(mws1gt_1,122,155,133).  visible(mws1gt_1,133). happensAt(walking(mws1gt_1),133). orientation(mws1gt_2,164,133). coords(mws1gt_2,29,209,133).  visible(mws1gt_2,133). happensAt(walking(mws1gt_2),133).
#show holdsAt/2.