
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
orientation(br4gt_0,112,130). coords(br4gt_0,306,257,130).  visible(br4gt_0,130). happensAt(walking(br4gt_0),130). orientation(br4gt_1,2,130). coords(br4gt_1,218,241,130).  visible(br4gt_1,130). happensAt(walking(br4gt_1),130).
orientation(br4gt_0,40,131). coords(br4gt_0,307,259,131).  visible(br4gt_0,131). happensAt(walking(br4gt_0),131). orientation(br4gt_1,1,131). coords(br4gt_1,218,239,131).  visible(br4gt_1,131). happensAt(walking(br4gt_1),131).
orientation(br4gt_0,59,132). coords(br4gt_0,308,260,132).  visible(br4gt_0,132). happensAt(walking(br4gt_0),132). orientation(br4gt_1,179,132). coords(br4gt_1,217,236,132).  visible(br4gt_1,132). happensAt(walking(br4gt_1),132).
orientation(br4gt_0,59,133). coords(br4gt_0,309,261,133).  visible(br4gt_0,133). happensAt(walking(br4gt_0),133). orientation(br4gt_1,179,133). coords(br4gt_1,216,232,133).  visible(br4gt_1,133). happensAt(walking(br4gt_1),133).
orientation(br4gt_0,66,134). coords(br4gt_0,310,263,134).  visible(br4gt_0,134). happensAt(walking(br4gt_0),134). orientation(br4gt_1,179,134). coords(br4gt_1,215,229,134).  visible(br4gt_1,134). happensAt(walking(br4gt_1),134).
orientation(br4gt_0,56,135). coords(br4gt_0,311,264,135).  visible(br4gt_0,135). happensAt(walking(br4gt_0),135). orientation(br4gt_1,178,135). coords(br4gt_1,215,227,135).  visible(br4gt_1,135). happensAt(walking(br4gt_1),135).
orientation(br4gt_0,48,136). coords(br4gt_0,311,265,136).  visible(br4gt_0,136). happensAt(walking(br4gt_0),136). orientation(br4gt_1,159,136). coords(br4gt_1,215,227,136).  visible(br4gt_1,136). happensAt(walking(br4gt_1),136).
orientation(br4gt_0,49,137). coords(br4gt_0,312,267,137).  visible(br4gt_0,137). happensAt(walking(br4gt_0),137). orientation(br4gt_1,161,137). coords(br4gt_1,214,223,137).  visible(br4gt_1,137). happensAt(walking(br4gt_1),137).
orientation(br4gt_0,49,138). coords(br4gt_0,313,269,138).  visible(br4gt_0,138). happensAt(walking(br4gt_0),138). orientation(br4gt_1,161,138). coords(br4gt_1,213,222,138).  visible(br4gt_1,138). happensAt(walking(br4gt_1),138).
orientation(br4gt_0,59,139). coords(br4gt_0,313,271,139).  visible(br4gt_0,139). happensAt(walking(br4gt_0),139). orientation(br4gt_1,164,139). coords(br4gt_1,212,220,139).  visible(br4gt_1,139). happensAt(walking(br4gt_1),139).
orientation(br4gt_0,66,140). coords(br4gt_0,314,273,140).  visible(br4gt_0,140). happensAt(walking(br4gt_0),140). orientation(br4gt_1,151,140). coords(br4gt_1,212,219,140).  visible(br4gt_1,140). happensAt(walking(br4gt_1),140).
orientation(br4gt_0,61,141). coords(br4gt_0,314,275,141).  visible(br4gt_0,141). happensAt(walking(br4gt_0),141). orientation(br4gt_1,151,141). coords(br4gt_1,211,217,141).  visible(br4gt_1,141). happensAt(walking(br4gt_1),141).
#show holdsAt/2.