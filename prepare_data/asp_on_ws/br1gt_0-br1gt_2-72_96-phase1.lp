
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

	orientation(br1gt_0,157,72). coords(br1gt_0,187,149,72).  visible(br1gt_0,72). happensAt(walking(br1gt_0),72). orientation(br1gt_2,171,72). coords(br1gt_2,86,207,72).  visible(br1gt_2,72). happensAt(inactive(br1gt_2),72).
orientation(br1gt_0,157,73). coords(br1gt_0,187,147,73).  visible(br1gt_0,73). happensAt(walking(br1gt_0),73). orientation(br1gt_2,171,73). coords(br1gt_2,86,207,73).  visible(br1gt_2,73). happensAt(inactive(br1gt_2),73).
orientation(br1gt_0,157,74). coords(br1gt_0,186,145,74).  visible(br1gt_0,74). happensAt(walking(br1gt_0),74). orientation(br1gt_2,171,74). coords(br1gt_2,86,207,74).  visible(br1gt_2,74). happensAt(inactive(br1gt_2),74).
orientation(br1gt_0,148,75). coords(br1gt_0,187,143,75).  visible(br1gt_0,75). happensAt(inactive(br1gt_0),75). orientation(br1gt_2,171,75). coords(br1gt_2,86,207,75).  visible(br1gt_2,75). happensAt(inactive(br1gt_2),75).
orientation(br1gt_0,151,76). coords(br1gt_0,187,142,76).  visible(br1gt_0,76). happensAt(inactive(br1gt_0),76). orientation(br1gt_2,171,76). coords(br1gt_2,86,207,76).  visible(br1gt_2,76). happensAt(inactive(br1gt_2),76).
orientation(br1gt_0,147,77). coords(br1gt_0,186,140,77).  visible(br1gt_0,77). happensAt(inactive(br1gt_0),77). orientation(br1gt_2,171,77). coords(br1gt_2,86,207,77).  visible(br1gt_2,77). happensAt(inactive(br1gt_2),77).
orientation(br1gt_0,146,78). coords(br1gt_0,185,139,78).  visible(br1gt_0,78). happensAt(inactive(br1gt_0),78). orientation(br1gt_2,171,78). coords(br1gt_2,86,207,78).  visible(br1gt_2,78). happensAt(inactive(br1gt_2),78).
orientation(br1gt_0,151,79). coords(br1gt_0,185,139,79).  visible(br1gt_0,79). happensAt(inactive(br1gt_0),79). orientation(br1gt_2,171,79). coords(br1gt_2,86,207,79).  visible(br1gt_2,79). happensAt(inactive(br1gt_2),79).
orientation(br1gt_0,153,80). coords(br1gt_0,184,141,80).  visible(br1gt_0,80). happensAt(inactive(br1gt_0),80). orientation(br1gt_2,171,80). coords(br1gt_2,86,207,80).  visible(br1gt_2,80). happensAt(inactive(br1gt_2),80).
orientation(br1gt_0,154,81). coords(br1gt_0,184,140,81).  visible(br1gt_0,81). happensAt(inactive(br1gt_0),81). orientation(br1gt_2,171,81). coords(br1gt_2,86,207,81).  visible(br1gt_2,81). happensAt(inactive(br1gt_2),81).
orientation(br1gt_0,155,82). coords(br1gt_0,184,141,82).  visible(br1gt_0,82). happensAt(inactive(br1gt_0),82). orientation(br1gt_2,171,82). coords(br1gt_2,86,207,82).  visible(br1gt_2,82). happensAt(inactive(br1gt_2),82).
orientation(br1gt_0,157,83). coords(br1gt_0,184,141,83).  visible(br1gt_0,83). happensAt(inactive(br1gt_0),83). orientation(br1gt_2,171,83). coords(br1gt_2,86,207,83).  visible(br1gt_2,83). happensAt(inactive(br1gt_2),83).
orientation(br1gt_0,157,84). coords(br1gt_0,183,141,84).  visible(br1gt_0,84). happensAt(inactive(br1gt_0),84). orientation(br1gt_2,171,84). coords(br1gt_2,86,207,84).  visible(br1gt_2,84). happensAt(inactive(br1gt_2),84).
orientation(br1gt_0,157,85). coords(br1gt_0,183,141,85).  visible(br1gt_0,85). happensAt(inactive(br1gt_0),85). orientation(br1gt_2,171,85). coords(br1gt_2,86,207,85).  visible(br1gt_2,85). happensAt(inactive(br1gt_2),85).
orientation(br1gt_0,157,86). coords(br1gt_0,183,141,86).  visible(br1gt_0,86). happensAt(inactive(br1gt_0),86). orientation(br1gt_2,171,86). coords(br1gt_2,86,207,86).  visible(br1gt_2,86). happensAt(inactive(br1gt_2),86).
orientation(br1gt_0,157,87). coords(br1gt_0,183,141,87).  visible(br1gt_0,87). happensAt(inactive(br1gt_0),87). orientation(br1gt_2,171,87). coords(br1gt_2,86,207,87).  visible(br1gt_2,87). happensAt(inactive(br1gt_2),87).
orientation(br1gt_0,157,88). coords(br1gt_0,183,141,88).  visible(br1gt_0,88). happensAt(inactive(br1gt_0),88). orientation(br1gt_2,171,88). coords(br1gt_2,86,207,88).  visible(br1gt_2,88). happensAt(inactive(br1gt_2),88).
orientation(br1gt_0,157,89). coords(br1gt_0,183,141,89).  visible(br1gt_0,89). happensAt(inactive(br1gt_0),89). orientation(br1gt_2,171,89). coords(br1gt_2,86,207,89).  visible(br1gt_2,89). happensAt(inactive(br1gt_2),89).
orientation(br1gt_0,157,90). coords(br1gt_0,183,141,90).  visible(br1gt_0,90). happensAt(inactive(br1gt_0),90). orientation(br1gt_2,171,90). coords(br1gt_2,86,207,90).  visible(br1gt_2,90). happensAt(inactive(br1gt_2),90).
orientation(br1gt_0,157,91). coords(br1gt_0,183,141,91).  visible(br1gt_0,91). happensAt(inactive(br1gt_0),91). orientation(br1gt_2,171,91). coords(br1gt_2,86,207,91).  visible(br1gt_2,91). happensAt(inactive(br1gt_2),91).
orientation(br1gt_0,157,92). coords(br1gt_0,183,141,92).  visible(br1gt_0,92). happensAt(inactive(br1gt_0),92). orientation(br1gt_2,171,92). coords(br1gt_2,86,207,92).  visible(br1gt_2,92). happensAt(inactive(br1gt_2),92).
orientation(br1gt_0,157,93). coords(br1gt_0,183,140,93).  visible(br1gt_0,93). happensAt(inactive(br1gt_0),93). orientation(br1gt_2,171,93). coords(br1gt_2,86,207,93).  visible(br1gt_2,93). happensAt(inactive(br1gt_2),93).
orientation(br1gt_0,157,94). coords(br1gt_0,183,140,94).  visible(br1gt_0,94). happensAt(inactive(br1gt_0),94). orientation(br1gt_2,171,94). coords(br1gt_2,86,207,94).  visible(br1gt_2,94). happensAt(inactive(br1gt_2),94).
orientation(br1gt_0,161,95). coords(br1gt_0,183,140,95).  visible(br1gt_0,95). happensAt(inactive(br1gt_0),95). orientation(br1gt_2,171,95). coords(br1gt_2,86,207,95).  visible(br1gt_2,95). happensAt(inactive(br1gt_2),95).
orientation(br1gt_0,159,96). coords(br1gt_0,183,140,96).  visible(br1gt_0,96). happensAt(inactive(br1gt_0),96). orientation(br1gt_2,171,96). coords(br1gt_2,86,207,96).  visible(br1gt_2,96). happensAt(inactive(br1gt_2),96).
#show holdsAt/2.