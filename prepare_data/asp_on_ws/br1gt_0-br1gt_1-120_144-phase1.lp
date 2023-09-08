
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

	orientation(br1gt_0,152,120). coords(br1gt_0,197,149,120).  visible(br1gt_0,120). happensAt(walking(br1gt_0),120). orientation(br1gt_1,164,120). coords(br1gt_1,62,163,120).  visible(br1gt_1,120). happensAt(walking(br1gt_1),120).
orientation(br1gt_0,152,121). coords(br1gt_0,198,150,121).  visible(br1gt_0,121). happensAt(walking(br1gt_0),121). orientation(br1gt_1,164,121). coords(br1gt_1,61,163,121).  visible(br1gt_1,121). happensAt(walking(br1gt_1),121).
orientation(br1gt_0,152,122). coords(br1gt_0,198,150,122).  visible(br1gt_0,122). happensAt(walking(br1gt_0),122). orientation(br1gt_1,164,122). coords(br1gt_1,60,163,122).  visible(br1gt_1,122). happensAt(walking(br1gt_1),122).
orientation(br1gt_0,152,123). coords(br1gt_0,199,153,123).  visible(br1gt_0,123). happensAt(walking(br1gt_0),123). orientation(br1gt_1,164,123). coords(br1gt_1,60,163,123).  visible(br1gt_1,123). happensAt(walking(br1gt_1),123).
orientation(br1gt_0,152,124). coords(br1gt_0,199,155,124).  visible(br1gt_0,124). happensAt(walking(br1gt_0),124). orientation(br1gt_1,164,124). coords(br1gt_1,59,162,124).  visible(br1gt_1,124). happensAt(walking(br1gt_1),124).
orientation(br1gt_0,152,125). coords(br1gt_0,200,157,125).  visible(br1gt_0,125). happensAt(walking(br1gt_0),125). orientation(br1gt_1,164,125). coords(br1gt_1,59,162,125).  visible(br1gt_1,125). happensAt(walking(br1gt_1),125).
orientation(br1gt_0,152,126). coords(br1gt_0,200,158,126).  visible(br1gt_0,126). happensAt(walking(br1gt_0),126). orientation(br1gt_1,164,126). coords(br1gt_1,59,162,126).  visible(br1gt_1,126). happensAt(walking(br1gt_1),126).
orientation(br1gt_0,152,127). coords(br1gt_0,202,162,127).  visible(br1gt_0,127). happensAt(walking(br1gt_0),127). orientation(br1gt_1,164,127). coords(br1gt_1,59,162,127).  visible(br1gt_1,127). happensAt(walking(br1gt_1),127).
orientation(br1gt_0,152,128). coords(br1gt_0,204,164,128).  visible(br1gt_0,128). happensAt(walking(br1gt_0),128). orientation(br1gt_1,164,128). coords(br1gt_1,59,162,128).  visible(br1gt_1,128). happensAt(walking(br1gt_1),128).
orientation(br1gt_0,152,129). coords(br1gt_0,207,167,129).  visible(br1gt_0,129). happensAt(walking(br1gt_0),129). orientation(br1gt_1,164,129). coords(br1gt_1,58,161,129).  visible(br1gt_1,129). happensAt(walking(br1gt_1),129).
orientation(br1gt_0,152,130). coords(br1gt_0,209,170,130).  visible(br1gt_0,130). happensAt(walking(br1gt_0),130). orientation(br1gt_1,164,130). coords(br1gt_1,58,161,130).  visible(br1gt_1,130). happensAt(walking(br1gt_1),130).
orientation(br1gt_0,152,131). coords(br1gt_0,213,174,131).  visible(br1gt_0,131). happensAt(walking(br1gt_0),131). orientation(br1gt_1,164,131). coords(br1gt_1,56,161,131).  visible(br1gt_1,131). happensAt(walking(br1gt_1),131).
orientation(br1gt_0,152,132). coords(br1gt_0,213,174,132).  visible(br1gt_0,132). happensAt(walking(br1gt_0),132). orientation(br1gt_1,164,132). coords(br1gt_1,56,161,132).  visible(br1gt_1,132). happensAt(walking(br1gt_1),132).
orientation(br1gt_0,152,133). coords(br1gt_0,216,175,133).  visible(br1gt_0,133). happensAt(walking(br1gt_0),133). orientation(br1gt_1,164,133). coords(br1gt_1,55,161,133).  visible(br1gt_1,133). happensAt(walking(br1gt_1),133).
orientation(br1gt_0,152,134). coords(br1gt_0,217,177,134).  visible(br1gt_0,134). happensAt(walking(br1gt_0),134). orientation(br1gt_1,164,134). coords(br1gt_1,55,161,134).  visible(br1gt_1,134). happensAt(walking(br1gt_1),134).
orientation(br1gt_0,152,135). coords(br1gt_0,218,178,135).  visible(br1gt_0,135). happensAt(walking(br1gt_0),135). orientation(br1gt_1,164,135). coords(br1gt_1,53,161,135).  visible(br1gt_1,135). happensAt(walking(br1gt_1),135).
orientation(br1gt_0,152,136). coords(br1gt_0,219,179,136).  visible(br1gt_0,136). happensAt(walking(br1gt_0),136). orientation(br1gt_1,164,136). coords(br1gt_1,53,161,136).  visible(br1gt_1,136). happensAt(walking(br1gt_1),136).
orientation(br1gt_0,152,137). coords(br1gt_0,220,179,137).  visible(br1gt_0,137). happensAt(walking(br1gt_0),137). orientation(br1gt_1,164,137). coords(br1gt_1,53,161,137).  visible(br1gt_1,137). happensAt(walking(br1gt_1),137).
orientation(br1gt_0,154,138). coords(br1gt_0,221,180,138).  visible(br1gt_0,138). happensAt(walking(br1gt_0),138). orientation(br1gt_1,164,138). coords(br1gt_1,53,161,138).  visible(br1gt_1,138). happensAt(walking(br1gt_1),138).
orientation(br1gt_0,154,139). coords(br1gt_0,222,181,139).  visible(br1gt_0,139). happensAt(walking(br1gt_0),139). orientation(br1gt_1,164,139). coords(br1gt_1,53,161,139).  visible(br1gt_1,139). happensAt(walking(br1gt_1),139).
orientation(br1gt_0,156,140). coords(br1gt_0,222,182,140).  visible(br1gt_0,140). happensAt(walking(br1gt_0),140). orientation(br1gt_1,164,140). coords(br1gt_1,53,161,140).  visible(br1gt_1,140). happensAt(walking(br1gt_1),140).
orientation(br1gt_0,159,141). coords(br1gt_0,226,183,141).  visible(br1gt_0,141). happensAt(walking(br1gt_0),141). orientation(br1gt_1,164,141). coords(br1gt_1,52,160,141).  visible(br1gt_1,141). happensAt(walking(br1gt_1),141).
orientation(br1gt_0,165,142). coords(br1gt_0,229,184,142).  visible(br1gt_0,142). happensAt(walking(br1gt_0),142). orientation(br1gt_1,164,142). coords(br1gt_1,52,160,142).  visible(br1gt_1,142). happensAt(walking(br1gt_1),142).
orientation(br1gt_0,165,143). coords(br1gt_0,232,185,143).  visible(br1gt_0,143). happensAt(walking(br1gt_0),143). orientation(br1gt_1,164,143). coords(br1gt_1,51,160,143).  visible(br1gt_1,143). happensAt(walking(br1gt_1),143).
orientation(br1gt_0,159,144). coords(br1gt_0,235,189,144).  visible(br1gt_0,144). happensAt(walking(br1gt_0),144). orientation(br1gt_1,164,144). coords(br1gt_1,51,160,144).  visible(br1gt_1,144). happensAt(walking(br1gt_1),144).
#show holdsAt/2.