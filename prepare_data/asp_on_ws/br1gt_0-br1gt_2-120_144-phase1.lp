
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

	orientation(br1gt_0,152,120). coords(br1gt_0,197,149,120).  visible(br1gt_0,120). happensAt(walking(br1gt_0),120). orientation(br1gt_2,171,120). coords(br1gt_2,86,207,120).  visible(br1gt_2,120). happensAt(inactive(br1gt_2),120).
orientation(br1gt_0,152,121). coords(br1gt_0,198,150,121).  visible(br1gt_0,121). happensAt(walking(br1gt_0),121). orientation(br1gt_2,171,121). coords(br1gt_2,86,207,121).  visible(br1gt_2,121). happensAt(inactive(br1gt_2),121).
orientation(br1gt_0,152,122). coords(br1gt_0,198,150,122).  visible(br1gt_0,122). happensAt(walking(br1gt_0),122). orientation(br1gt_2,171,122). coords(br1gt_2,86,207,122).  visible(br1gt_2,122). happensAt(inactive(br1gt_2),122).
orientation(br1gt_0,152,123). coords(br1gt_0,199,153,123).  visible(br1gt_0,123). happensAt(walking(br1gt_0),123). orientation(br1gt_2,171,123). coords(br1gt_2,86,207,123).  visible(br1gt_2,123). happensAt(inactive(br1gt_2),123).
orientation(br1gt_0,152,124). coords(br1gt_0,199,155,124).  visible(br1gt_0,124). happensAt(walking(br1gt_0),124). orientation(br1gt_2,171,124). coords(br1gt_2,86,207,124).  visible(br1gt_2,124). happensAt(inactive(br1gt_2),124).
orientation(br1gt_0,152,125). coords(br1gt_0,200,157,125).  visible(br1gt_0,125). happensAt(walking(br1gt_0),125). orientation(br1gt_2,171,125). coords(br1gt_2,86,207,125).  visible(br1gt_2,125). happensAt(inactive(br1gt_2),125).
orientation(br1gt_0,152,126). coords(br1gt_0,200,158,126).  visible(br1gt_0,126). happensAt(walking(br1gt_0),126). orientation(br1gt_2,171,126). coords(br1gt_2,86,207,126).  visible(br1gt_2,126). happensAt(inactive(br1gt_2),126).
orientation(br1gt_0,152,127). coords(br1gt_0,202,162,127).  visible(br1gt_0,127). happensAt(walking(br1gt_0),127). orientation(br1gt_2,171,127). coords(br1gt_2,86,207,127).  visible(br1gt_2,127). happensAt(inactive(br1gt_2),127).
orientation(br1gt_0,152,128). coords(br1gt_0,204,164,128).  visible(br1gt_0,128). happensAt(walking(br1gt_0),128). orientation(br1gt_2,171,128). coords(br1gt_2,86,207,128).  visible(br1gt_2,128). happensAt(inactive(br1gt_2),128).
orientation(br1gt_0,152,129). coords(br1gt_0,207,167,129).  visible(br1gt_0,129). happensAt(walking(br1gt_0),129). orientation(br1gt_2,171,129). coords(br1gt_2,86,207,129).  visible(br1gt_2,129). happensAt(inactive(br1gt_2),129).
orientation(br1gt_0,152,130). coords(br1gt_0,209,170,130).  visible(br1gt_0,130). happensAt(walking(br1gt_0),130). orientation(br1gt_2,171,130). coords(br1gt_2,86,207,130).  visible(br1gt_2,130). happensAt(inactive(br1gt_2),130).
orientation(br1gt_0,152,131). coords(br1gt_0,213,174,131).  visible(br1gt_0,131). happensAt(walking(br1gt_0),131). orientation(br1gt_2,171,131). coords(br1gt_2,86,207,131).  visible(br1gt_2,131). happensAt(inactive(br1gt_2),131).
orientation(br1gt_0,152,132). coords(br1gt_0,213,174,132).  visible(br1gt_0,132). happensAt(walking(br1gt_0),132). orientation(br1gt_2,171,132). coords(br1gt_2,86,207,132).  visible(br1gt_2,132). happensAt(inactive(br1gt_2),132).
orientation(br1gt_0,152,133). coords(br1gt_0,216,175,133).  visible(br1gt_0,133). happensAt(walking(br1gt_0),133). orientation(br1gt_2,171,133). coords(br1gt_2,86,207,133).  visible(br1gt_2,133). happensAt(inactive(br1gt_2),133).
orientation(br1gt_0,152,134). coords(br1gt_0,217,177,134).  visible(br1gt_0,134). happensAt(walking(br1gt_0),134). orientation(br1gt_2,171,134). coords(br1gt_2,86,207,134).  visible(br1gt_2,134). happensAt(inactive(br1gt_2),134).
orientation(br1gt_0,152,135). coords(br1gt_0,218,178,135).  visible(br1gt_0,135). happensAt(walking(br1gt_0),135). orientation(br1gt_2,171,135). coords(br1gt_2,86,207,135).  visible(br1gt_2,135). happensAt(inactive(br1gt_2),135).
orientation(br1gt_0,152,136). coords(br1gt_0,219,179,136).  visible(br1gt_0,136). happensAt(walking(br1gt_0),136). orientation(br1gt_2,171,136). coords(br1gt_2,86,207,136).  visible(br1gt_2,136). happensAt(inactive(br1gt_2),136).
orientation(br1gt_0,152,137). coords(br1gt_0,220,179,137).  visible(br1gt_0,137). happensAt(walking(br1gt_0),137). orientation(br1gt_2,171,137). coords(br1gt_2,86,207,137).  visible(br1gt_2,137). happensAt(inactive(br1gt_2),137).
orientation(br1gt_0,154,138). coords(br1gt_0,221,180,138).  visible(br1gt_0,138). happensAt(walking(br1gt_0),138). orientation(br1gt_2,171,138). coords(br1gt_2,86,207,138).  visible(br1gt_2,138). happensAt(inactive(br1gt_2),138).
orientation(br1gt_0,154,139). coords(br1gt_0,222,181,139).  visible(br1gt_0,139). happensAt(walking(br1gt_0),139). orientation(br1gt_2,171,139). coords(br1gt_2,86,207,139).  visible(br1gt_2,139). happensAt(inactive(br1gt_2),139).
orientation(br1gt_0,156,140). coords(br1gt_0,222,182,140).  visible(br1gt_0,140). happensAt(walking(br1gt_0),140). orientation(br1gt_2,171,140). coords(br1gt_2,86,207,140).  visible(br1gt_2,140). happensAt(inactive(br1gt_2),140).
orientation(br1gt_0,159,141). coords(br1gt_0,226,183,141).  visible(br1gt_0,141). happensAt(walking(br1gt_0),141). orientation(br1gt_2,171,141). coords(br1gt_2,86,207,141).  visible(br1gt_2,141). happensAt(inactive(br1gt_2),141).
orientation(br1gt_0,165,142). coords(br1gt_0,229,184,142).  visible(br1gt_0,142). happensAt(walking(br1gt_0),142). orientation(br1gt_2,171,142). coords(br1gt_2,86,207,142).  visible(br1gt_2,142). happensAt(inactive(br1gt_2),142).
orientation(br1gt_0,165,143). coords(br1gt_0,232,185,143).  visible(br1gt_0,143). happensAt(walking(br1gt_0),143). orientation(br1gt_2,171,143). coords(br1gt_2,86,207,143).  visible(br1gt_2,143). happensAt(inactive(br1gt_2),143).
orientation(br1gt_0,159,144). coords(br1gt_0,235,189,144).  visible(br1gt_0,144). happensAt(walking(br1gt_0),144). orientation(br1gt_2,171,144). coords(br1gt_2,86,207,144).  visible(br1gt_2,144). happensAt(inactive(br1gt_2),144).
#show holdsAt/2.