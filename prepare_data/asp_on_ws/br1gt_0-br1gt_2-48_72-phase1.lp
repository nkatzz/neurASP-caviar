
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

	orientation(br1gt_0,158,48). coords(br1gt_0,209,180,48).  visible(br1gt_0,48). happensAt(walking(br1gt_0),48). orientation(br1gt_2,171,48). coords(br1gt_2,86,205,48).  visible(br1gt_2,48). happensAt(inactive(br1gt_2),48).
orientation(br1gt_0,154,49). coords(br1gt_0,208,178,49).  visible(br1gt_0,49). happensAt(walking(br1gt_0),49). orientation(br1gt_2,171,49). coords(br1gt_2,86,205,49).  visible(br1gt_2,49). happensAt(inactive(br1gt_2),49).
orientation(br1gt_0,157,50). coords(br1gt_0,207,176,50).  visible(br1gt_0,50). happensAt(walking(br1gt_0),50). orientation(br1gt_2,171,50). coords(br1gt_2,86,205,50).  visible(br1gt_2,50). happensAt(inactive(br1gt_2),50).
orientation(br1gt_0,157,51). coords(br1gt_0,205,174,51).  visible(br1gt_0,51). happensAt(walking(br1gt_0),51). orientation(br1gt_2,171,51). coords(br1gt_2,86,206,51).  visible(br1gt_2,51). happensAt(inactive(br1gt_2),51).
orientation(br1gt_0,157,52). coords(br1gt_0,202,173,52).  visible(br1gt_0,52). happensAt(walking(br1gt_0),52). orientation(br1gt_2,171,52). coords(br1gt_2,86,207,52).  visible(br1gt_2,52). happensAt(inactive(br1gt_2),52).
orientation(br1gt_0,157,53). coords(br1gt_0,198,171,53).  visible(br1gt_0,53). happensAt(walking(br1gt_0),53). orientation(br1gt_2,171,53). coords(br1gt_2,86,206,53).  visible(br1gt_2,53). happensAt(inactive(br1gt_2),53).
orientation(br1gt_0,161,54). coords(br1gt_0,196,168,54).  visible(br1gt_0,54). happensAt(walking(br1gt_0),54). orientation(br1gt_2,171,54). coords(br1gt_2,86,206,54).  visible(br1gt_2,54). happensAt(inactive(br1gt_2),54).
orientation(br1gt_0,161,55). coords(br1gt_0,192,165,55).  visible(br1gt_0,55). happensAt(walking(br1gt_0),55). orientation(br1gt_2,171,55). coords(br1gt_2,86,206,55).  visible(br1gt_2,55). happensAt(inactive(br1gt_2),55).
orientation(br1gt_0,161,56). coords(br1gt_0,190,163,56).  visible(br1gt_0,56). happensAt(walking(br1gt_0),56). orientation(br1gt_2,171,56). coords(br1gt_2,86,207,56).  visible(br1gt_2,56). happensAt(inactive(br1gt_2),56).
orientation(br1gt_0,161,57). coords(br1gt_0,187,161,57).  visible(br1gt_0,57). happensAt(walking(br1gt_0),57). orientation(br1gt_2,171,57). coords(br1gt_2,86,207,57).  visible(br1gt_2,57). happensAt(inactive(br1gt_2),57).
orientation(br1gt_0,161,58). coords(br1gt_0,186,161,58).  visible(br1gt_0,58). happensAt(walking(br1gt_0),58). orientation(br1gt_2,171,58). coords(br1gt_2,86,207,58).  visible(br1gt_2,58). happensAt(inactive(br1gt_2),58).
orientation(br1gt_0,161,59). coords(br1gt_0,185,160,59).  visible(br1gt_0,59). happensAt(walking(br1gt_0),59). orientation(br1gt_2,171,59). coords(br1gt_2,86,207,59).  visible(br1gt_2,59). happensAt(inactive(br1gt_2),59).
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
#show holdsAt/2.