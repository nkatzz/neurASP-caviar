
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

	orientation(wk1gt_0,131,123). coords(wk1gt_0,212,133,123).  visible(wk1gt_0,123). happensAt(active(wk1gt_0),123). orientation(wk1gt_4,0,123). coords(wk1gt_4,45,186,123).  visible(wk1gt_4,123). happensAt(walking(wk1gt_4),123).
orientation(wk1gt_0,127,124). coords(wk1gt_0,213,135,124).  visible(wk1gt_0,124). happensAt(active(wk1gt_0),124). orientation(wk1gt_4,0,124). coords(wk1gt_4,45,186,124).  visible(wk1gt_4,124). happensAt(walking(wk1gt_4),124).
orientation(wk1gt_0,131,125). coords(wk1gt_0,214,138,125).  visible(wk1gt_0,125). happensAt(active(wk1gt_0),125). orientation(wk1gt_4,0,125). coords(wk1gt_4,45,186,125).  visible(wk1gt_4,125). happensAt(walking(wk1gt_4),125).
orientation(wk1gt_0,129,126). coords(wk1gt_0,216,140,126).  visible(wk1gt_0,126). happensAt(active(wk1gt_0),126). orientation(wk1gt_4,0,126). coords(wk1gt_4,45,186,126).  visible(wk1gt_4,126). happensAt(walking(wk1gt_4),126).
orientation(wk1gt_0,129,127). coords(wk1gt_0,218,144,127).  visible(wk1gt_0,127). happensAt(walking(wk1gt_0),127). orientation(wk1gt_4,0,127). coords(wk1gt_4,45,186,127).  visible(wk1gt_4,127). happensAt(walking(wk1gt_4),127).
orientation(wk1gt_0,125,128). coords(wk1gt_0,220,146,128).  visible(wk1gt_0,128). happensAt(walking(wk1gt_0),128). orientation(wk1gt_4,0,128). coords(wk1gt_4,45,186,128).  visible(wk1gt_4,128). happensAt(walking(wk1gt_4),128).
orientation(wk1gt_0,125,129). coords(wk1gt_0,222,149,129).  visible(wk1gt_0,129). happensAt(walking(wk1gt_0),129). orientation(wk1gt_4,0,129). coords(wk1gt_4,45,186,129).  visible(wk1gt_4,129). happensAt(walking(wk1gt_4),129).
orientation(wk1gt_0,125,130). coords(wk1gt_0,224,150,130).  visible(wk1gt_0,130). happensAt(walking(wk1gt_0),130). orientation(wk1gt_4,0,130). coords(wk1gt_4,45,186,130).  visible(wk1gt_4,130). happensAt(walking(wk1gt_4),130).
orientation(wk1gt_0,135,131). coords(wk1gt_0,225,152,131).  visible(wk1gt_0,131). happensAt(walking(wk1gt_0),131). orientation(wk1gt_4,0,131). coords(wk1gt_4,45,186,131).  visible(wk1gt_4,131). happensAt(walking(wk1gt_4),131).
orientation(wk1gt_0,135,132). coords(wk1gt_0,226,153,132).  visible(wk1gt_0,132). happensAt(walking(wk1gt_0),132). orientation(wk1gt_4,0,132). coords(wk1gt_4,45,186,132).  visible(wk1gt_4,132). happensAt(walking(wk1gt_4),132).
orientation(wk1gt_0,144,133). coords(wk1gt_0,227,153,133).  visible(wk1gt_0,133). happensAt(walking(wk1gt_0),133). orientation(wk1gt_4,0,133). coords(wk1gt_4,45,185,133).  visible(wk1gt_4,133). happensAt(walking(wk1gt_4),133).
orientation(wk1gt_0,155,134). coords(wk1gt_0,228,154,134).  visible(wk1gt_0,134). happensAt(walking(wk1gt_0),134). orientation(wk1gt_4,0,134). coords(wk1gt_4,45,185,134).  visible(wk1gt_4,134). happensAt(walking(wk1gt_4),134).
orientation(wk1gt_0,165,135). coords(wk1gt_0,229,156,135).  visible(wk1gt_0,135). happensAt(walking(wk1gt_0),135). orientation(wk1gt_4,0,135). coords(wk1gt_4,45,185,135).  visible(wk1gt_4,135). happensAt(walking(wk1gt_4),135).
orientation(wk1gt_0,165,136). coords(wk1gt_0,229,158,136).  visible(wk1gt_0,136). happensAt(walking(wk1gt_0),136). orientation(wk1gt_4,0,136). coords(wk1gt_4,45,185,136).  visible(wk1gt_4,136). happensAt(walking(wk1gt_4),136).
orientation(wk1gt_0,165,137). coords(wk1gt_0,232,159,137).  visible(wk1gt_0,137). happensAt(walking(wk1gt_0),137). orientation(wk1gt_4,0,137). coords(wk1gt_4,45,184,137).  visible(wk1gt_4,137). happensAt(walking(wk1gt_4),137).
orientation(wk1gt_0,178,138). coords(wk1gt_0,234,162,138).  visible(wk1gt_0,138). happensAt(walking(wk1gt_0),138). orientation(wk1gt_4,0,138). coords(wk1gt_4,45,184,138).  visible(wk1gt_4,138). happensAt(walking(wk1gt_4),138).
orientation(wk1gt_0,7,139). coords(wk1gt_0,237,164,139).  visible(wk1gt_0,139). happensAt(walking(wk1gt_0),139). orientation(wk1gt_4,0,139). coords(wk1gt_4,45,184,139).  visible(wk1gt_4,139). happensAt(walking(wk1gt_4),139).
orientation(wk1gt_0,10,140). coords(wk1gt_0,240,167,140).  visible(wk1gt_0,140). happensAt(walking(wk1gt_0),140). orientation(wk1gt_4,0,140). coords(wk1gt_4,45,184,140).  visible(wk1gt_4,140). happensAt(walking(wk1gt_4),140).
orientation(wk1gt_0,2,141). coords(wk1gt_0,242,172,141).  visible(wk1gt_0,141). happensAt(walking(wk1gt_0),141). orientation(wk1gt_4,0,141). coords(wk1gt_4,45,184,141).  visible(wk1gt_4,141). happensAt(walking(wk1gt_4),141).
orientation(wk1gt_0,171,142). coords(wk1gt_0,244,178,142).  visible(wk1gt_0,142). happensAt(walking(wk1gt_0),142). orientation(wk1gt_4,0,142). coords(wk1gt_4,46,184,142).  visible(wk1gt_4,142). happensAt(walking(wk1gt_4),142).
orientation(wk1gt_0,170,143). coords(wk1gt_0,245,181,143).  visible(wk1gt_0,143). happensAt(walking(wk1gt_0),143). orientation(wk1gt_4,0,143). coords(wk1gt_4,46,184,143).  visible(wk1gt_4,143). happensAt(walking(wk1gt_4),143).
orientation(wk1gt_0,167,144). coords(wk1gt_0,247,185,144).  visible(wk1gt_0,144). happensAt(walking(wk1gt_0),144). orientation(wk1gt_4,0,144). coords(wk1gt_4,46,184,144).  visible(wk1gt_4,144). happensAt(walking(wk1gt_4),144).
orientation(wk1gt_0,175,145). coords(wk1gt_0,248,188,145).  visible(wk1gt_0,145). happensAt(walking(wk1gt_0),145). orientation(wk1gt_4,0,145). coords(wk1gt_4,46,184,145).  visible(wk1gt_4,145). happensAt(walking(wk1gt_4),145).
orientation(wk1gt_0,175,146). coords(wk1gt_0,249,191,146).  visible(wk1gt_0,146). happensAt(walking(wk1gt_0),146). orientation(wk1gt_4,0,146). coords(wk1gt_4,46,184,146).  visible(wk1gt_4,146). happensAt(walking(wk1gt_4),146).
orientation(wk1gt_0,179,147). coords(wk1gt_0,251,195,147).  visible(wk1gt_0,147). happensAt(walking(wk1gt_0),147). orientation(wk1gt_4,0,147). coords(wk1gt_4,46,183,147).  visible(wk1gt_4,147). happensAt(walking(wk1gt_4),147).
#show holdsAt/2.