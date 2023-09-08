
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

	orientation(br1gt_0,158,60). coords(br1gt_0,183,159,60).  visible(br1gt_0,60). happensAt(walking(br1gt_0),60). orientation(br1gt_2,171,60). coords(br1gt_2,86,207,60).  visible(br1gt_2,60). happensAt(inactive(br1gt_2),60).
orientation(br1gt_0,158,61). coords(br1gt_0,183,159,61).  visible(br1gt_0,61). happensAt(walking(br1gt_0),61). orientation(br1gt_2,171,61). coords(br1gt_2,86,207,61).  visible(br1gt_2,61). happensAt(inactive(br1gt_2),61).
orientation(br1gt_0,158,62). coords(br1gt_0,183,159,62).  visible(br1gt_0,62). happensAt(walking(br1gt_0),62). orientation(br1gt_2,171,62). coords(br1gt_2,86,207,62).  visible(br1gt_2,62). happensAt(inactive(br1gt_2),62).
orientation(br1gt_0,158,63). coords(br1gt_0,184,158,63).  visible(br1gt_0,63). happensAt(walking(br1gt_0),63). orientation(br1gt_2,171,63). coords(br1gt_2,86,207,63).  visible(br1gt_2,63). happensAt(inactive(br1gt_2),63).
orientation(br1gt_0,157,64). coords(br1gt_0,184,158,64).  visible(br1gt_0,64). happensAt(walking(br1gt_0),64). orientation(br1gt_2,171,64). coords(br1gt_2,86,207,64).  visible(br1gt_2,64). happensAt(inactive(br1gt_2),64).
orientation(br1gt_0,157,65). coords(br1gt_0,184,157,65).  visible(br1gt_0,65). happensAt(walking(br1gt_0),65). orientation(br1gt_2,171,65). coords(br1gt_2,86,207,65).  visible(br1gt_2,65). happensAt(inactive(br1gt_2),65).
orientation(br1gt_0,157,66). coords(br1gt_0,185,156,66).  visible(br1gt_0,66). happensAt(walking(br1gt_0),66). orientation(br1gt_2,171,66). coords(br1gt_2,86,207,66).  visible(br1gt_2,66). happensAt(inactive(br1gt_2),66).
orientation(br1gt_0,157,67). coords(br1gt_0,186,154,67).  visible(br1gt_0,67). happensAt(walking(br1gt_0),67). orientation(br1gt_2,171,67). coords(br1gt_2,86,207,67).  visible(br1gt_2,67). happensAt(inactive(br1gt_2),67).
orientation(br1gt_0,157,68). coords(br1gt_0,187,154,68).  visible(br1gt_0,68). happensAt(walking(br1gt_0),68). orientation(br1gt_2,171,68). coords(br1gt_2,86,207,68).  visible(br1gt_2,68). happensAt(inactive(br1gt_2),68).
orientation(br1gt_0,157,69). coords(br1gt_0,187,152,69).  visible(br1gt_0,69). happensAt(walking(br1gt_0),69). orientation(br1gt_2,171,69). coords(br1gt_2,86,207,69).  visible(br1gt_2,69). happensAt(inactive(br1gt_2),69).
orientation(br1gt_0,157,70). coords(br1gt_0,187,151,70).  visible(br1gt_0,70). happensAt(walking(br1gt_0),70). orientation(br1gt_2,171,70). coords(br1gt_2,86,207,70).  visible(br1gt_2,70). happensAt(inactive(br1gt_2),70).
orientation(br1gt_0,157,71). coords(br1gt_0,188,151,71).  visible(br1gt_0,71). happensAt(walking(br1gt_0),71). orientation(br1gt_2,171,71). coords(br1gt_2,86,207,71).  visible(br1gt_2,71). happensAt(inactive(br1gt_2),71).
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
#show holdsAt/2.