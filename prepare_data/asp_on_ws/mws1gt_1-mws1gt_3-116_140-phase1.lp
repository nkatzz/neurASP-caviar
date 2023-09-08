
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
orientation(mws1gt_1,155,129). coords(mws1gt_1,115,159,129).  visible(mws1gt_1,129). happensAt(walking(mws1gt_1),129). orientation(mws1gt_3,150,129). coords(mws1gt_3,206,180,129).  visible(mws1gt_3,129). happensAt(walking(mws1gt_3),129).
orientation(mws1gt_1,155,130). coords(mws1gt_1,115,157,130).  visible(mws1gt_1,130). happensAt(walking(mws1gt_1),130). orientation(mws1gt_3,154,130). coords(mws1gt_3,205,179,130).  visible(mws1gt_3,130). happensAt(walking(mws1gt_3),130).
orientation(mws1gt_1,155,131). coords(mws1gt_1,118,156,131).  visible(mws1gt_1,131). happensAt(walking(mws1gt_1),131). orientation(mws1gt_3,154,131). coords(mws1gt_3,205,179,131).  visible(mws1gt_3,131). happensAt(walking(mws1gt_3),131).
orientation(mws1gt_1,155,132). coords(mws1gt_1,120,156,132).  visible(mws1gt_1,132). happensAt(walking(mws1gt_1),132). orientation(mws1gt_3,154,132). coords(mws1gt_3,204,178,132).  visible(mws1gt_3,132). happensAt(walking(mws1gt_3),132).
orientation(mws1gt_1,155,133). coords(mws1gt_1,122,155,133).  visible(mws1gt_1,133). happensAt(walking(mws1gt_1),133). orientation(mws1gt_3,154,133). coords(mws1gt_3,203,177,133).  visible(mws1gt_3,133). happensAt(walking(mws1gt_3),133).
orientation(mws1gt_1,155,134). coords(mws1gt_1,125,155,134).  visible(mws1gt_1,134). happensAt(walking(mws1gt_1),134). orientation(mws1gt_3,154,134). coords(mws1gt_3,203,177,134).  visible(mws1gt_3,134). happensAt(walking(mws1gt_3),134).
orientation(mws1gt_1,155,135). coords(mws1gt_1,125,154,135).  visible(mws1gt_1,135). happensAt(walking(mws1gt_1),135). orientation(mws1gt_3,154,135). coords(mws1gt_3,202,176,135).  visible(mws1gt_3,135). happensAt(walking(mws1gt_3),135).
orientation(mws1gt_1,155,136). coords(mws1gt_1,127,154,136).  visible(mws1gt_1,136). happensAt(walking(mws1gt_1),136). orientation(mws1gt_3,154,136). coords(mws1gt_3,201,175,136).  visible(mws1gt_3,136). happensAt(walking(mws1gt_3),136).
orientation(mws1gt_1,155,137). coords(mws1gt_1,128,154,137).  visible(mws1gt_1,137). happensAt(walking(mws1gt_1),137). orientation(mws1gt_3,154,137). coords(mws1gt_3,199,173,137).  visible(mws1gt_3,137). happensAt(walking(mws1gt_3),137).
orientation(mws1gt_1,155,138). coords(mws1gt_1,129,153,138).  visible(mws1gt_1,138). happensAt(walking(mws1gt_1),138). orientation(mws1gt_3,154,138). coords(mws1gt_3,197,171,138).  visible(mws1gt_3,138). happensAt(walking(mws1gt_3),138).
orientation(mws1gt_1,155,139). coords(mws1gt_1,129,153,139).  visible(mws1gt_1,139). happensAt(walking(mws1gt_1),139). orientation(mws1gt_3,154,139). coords(mws1gt_3,195,170,139).  visible(mws1gt_3,139). happensAt(walking(mws1gt_3),139).
orientation(mws1gt_1,155,140). coords(mws1gt_1,129,153,140).  visible(mws1gt_1,140). happensAt(walking(mws1gt_1),140). orientation(mws1gt_3,154,140). coords(mws1gt_3,193,168,140).  visible(mws1gt_3,140). happensAt(walking(mws1gt_3),140).
#show holdsAt/2.