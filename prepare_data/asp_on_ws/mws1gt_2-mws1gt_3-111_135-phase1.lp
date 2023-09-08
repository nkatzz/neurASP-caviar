
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

	orientation(mws1gt_2,164,111). coords(mws1gt_2,34,198,111).  visible(mws1gt_2,111). happensAt(walking(mws1gt_2),111). orientation(mws1gt_3,178,111). coords(mws1gt_3,228,211,111).  visible(mws1gt_3,111). happensAt(walking(mws1gt_3),111).
orientation(mws1gt_2,164,112). coords(mws1gt_2,34,198,112).  visible(mws1gt_2,112). happensAt(walking(mws1gt_2),112). orientation(mws1gt_3,178,112). coords(mws1gt_3,226,209,112).  visible(mws1gt_3,112). happensAt(walking(mws1gt_3),112).
orientation(mws1gt_2,164,113). coords(mws1gt_2,34,200,113).  visible(mws1gt_2,113). happensAt(walking(mws1gt_2),113). orientation(mws1gt_3,176,113). coords(mws1gt_3,224,206,113).  visible(mws1gt_3,113). happensAt(walking(mws1gt_3),113).
orientation(mws1gt_2,164,114). coords(mws1gt_2,34,200,114).  visible(mws1gt_2,114). happensAt(walking(mws1gt_2),114). orientation(mws1gt_3,178,114). coords(mws1gt_3,222,205,114).  visible(mws1gt_3,114). happensAt(walking(mws1gt_3),114).
orientation(mws1gt_2,164,115). coords(mws1gt_2,34,201,115).  visible(mws1gt_2,115). happensAt(walking(mws1gt_2),115). orientation(mws1gt_3,170,115). coords(mws1gt_3,221,203,115).  visible(mws1gt_3,115). happensAt(walking(mws1gt_3),115).
orientation(mws1gt_2,164,116). coords(mws1gt_2,34,202,116).  visible(mws1gt_2,116). happensAt(walking(mws1gt_2),116). orientation(mws1gt_3,160,116). coords(mws1gt_3,220,201,116).  visible(mws1gt_3,116). happensAt(walking(mws1gt_3),116).
orientation(mws1gt_2,164,117). coords(mws1gt_2,34,203,117).  visible(mws1gt_2,117). happensAt(walking(mws1gt_2),117). orientation(mws1gt_3,158,117). coords(mws1gt_3,219,199,117).  visible(mws1gt_3,117). happensAt(walking(mws1gt_3),117).
orientation(mws1gt_2,164,118). coords(mws1gt_2,34,203,118).  visible(mws1gt_2,118). happensAt(walking(mws1gt_2),118). orientation(mws1gt_3,160,118). coords(mws1gt_3,219,198,118).  visible(mws1gt_3,118). happensAt(walking(mws1gt_3),118).
orientation(mws1gt_2,164,119). coords(mws1gt_2,34,204,119).  visible(mws1gt_2,119). happensAt(walking(mws1gt_2),119). orientation(mws1gt_3,157,119). coords(mws1gt_3,219,197,119).  visible(mws1gt_3,119). happensAt(walking(mws1gt_3),119).
orientation(mws1gt_2,164,120). coords(mws1gt_2,34,205,120).  visible(mws1gt_2,120). happensAt(walking(mws1gt_2),120). orientation(mws1gt_3,150,120). coords(mws1gt_3,219,197,120).  visible(mws1gt_3,120). happensAt(walking(mws1gt_3),120).
orientation(mws1gt_2,164,121). coords(mws1gt_2,34,205,121).  visible(mws1gt_2,121). happensAt(walking(mws1gt_2),121). orientation(mws1gt_3,150,121). coords(mws1gt_3,218,197,121).  visible(mws1gt_3,121). happensAt(walking(mws1gt_3),121).
orientation(mws1gt_2,164,122). coords(mws1gt_2,33,205,122).  visible(mws1gt_2,122). happensAt(walking(mws1gt_2),122). orientation(mws1gt_3,150,122). coords(mws1gt_3,216,196,122).  visible(mws1gt_3,122). happensAt(walking(mws1gt_3),122).
orientation(mws1gt_2,164,123). coords(mws1gt_2,33,205,123).  visible(mws1gt_2,123). happensAt(walking(mws1gt_2),123). orientation(mws1gt_3,150,123). coords(mws1gt_3,214,194,123).  visible(mws1gt_3,123). happensAt(walking(mws1gt_3),123).
orientation(mws1gt_2,164,124). coords(mws1gt_2,33,206,124).  visible(mws1gt_2,124). happensAt(walking(mws1gt_2),124). orientation(mws1gt_3,150,124). coords(mws1gt_3,212,191,124).  visible(mws1gt_3,124). happensAt(walking(mws1gt_3),124).
orientation(mws1gt_2,164,125). coords(mws1gt_2,32,206,125).  visible(mws1gt_2,125). happensAt(walking(mws1gt_2),125). orientation(mws1gt_3,150,125). coords(mws1gt_3,210,189,125).  visible(mws1gt_3,125). happensAt(walking(mws1gt_3),125).
orientation(mws1gt_2,164,126). coords(mws1gt_2,32,206,126).  visible(mws1gt_2,126). happensAt(walking(mws1gt_2),126). orientation(mws1gt_3,150,126). coords(mws1gt_3,208,187,126).  visible(mws1gt_3,126). happensAt(walking(mws1gt_3),126).
orientation(mws1gt_2,164,127). coords(mws1gt_2,32,207,127).  visible(mws1gt_2,127). happensAt(walking(mws1gt_2),127). orientation(mws1gt_3,150,127). coords(mws1gt_3,207,184,127).  visible(mws1gt_3,127). happensAt(walking(mws1gt_3),127).
orientation(mws1gt_2,164,128). coords(mws1gt_2,32,207,128).  visible(mws1gt_2,128). happensAt(walking(mws1gt_2),128). orientation(mws1gt_3,150,128). coords(mws1gt_3,206,182,128).  visible(mws1gt_3,128). happensAt(walking(mws1gt_3),128).
orientation(mws1gt_2,164,129). coords(mws1gt_2,29,207,129).  visible(mws1gt_2,129). happensAt(walking(mws1gt_2),129). orientation(mws1gt_3,150,129). coords(mws1gt_3,206,180,129).  visible(mws1gt_3,129). happensAt(walking(mws1gt_3),129).
orientation(mws1gt_2,164,130). coords(mws1gt_2,29,208,130).  visible(mws1gt_2,130). happensAt(walking(mws1gt_2),130). orientation(mws1gt_3,154,130). coords(mws1gt_3,205,179,130).  visible(mws1gt_3,130). happensAt(walking(mws1gt_3),130).
orientation(mws1gt_2,164,131). coords(mws1gt_2,29,208,131).  visible(mws1gt_2,131). happensAt(walking(mws1gt_2),131). orientation(mws1gt_3,154,131). coords(mws1gt_3,205,179,131).  visible(mws1gt_3,131). happensAt(walking(mws1gt_3),131).
orientation(mws1gt_2,164,132). coords(mws1gt_2,29,209,132).  visible(mws1gt_2,132). happensAt(walking(mws1gt_2),132). orientation(mws1gt_3,154,132). coords(mws1gt_3,204,178,132).  visible(mws1gt_3,132). happensAt(walking(mws1gt_3),132).
orientation(mws1gt_2,164,133). coords(mws1gt_2,29,209,133).  visible(mws1gt_2,133). happensAt(walking(mws1gt_2),133). orientation(mws1gt_3,154,133). coords(mws1gt_3,203,177,133).  visible(mws1gt_3,133). happensAt(walking(mws1gt_3),133).
orientation(mws1gt_2,164,134). coords(mws1gt_2,29,210,134).  visible(mws1gt_2,134). happensAt(walking(mws1gt_2),134). orientation(mws1gt_3,154,134). coords(mws1gt_3,203,177,134).  visible(mws1gt_3,134). happensAt(walking(mws1gt_3),134).
orientation(mws1gt_2,164,135). coords(mws1gt_2,29,210,135).  happensAt(disappear(mws1gt_2),135). happensAt(walking(mws1gt_2),135). orientation(mws1gt_3,154,135). coords(mws1gt_3,202,176,135).  visible(mws1gt_3,135). happensAt(walking(mws1gt_3),135).
#show holdsAt/2.